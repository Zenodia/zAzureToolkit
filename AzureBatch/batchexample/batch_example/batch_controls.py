import datetime
import sys
import time
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels
import azure.storage.blob as azureblob

def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
    """
    Returns when all tasks in the specified job reach the Completed state.
    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be to monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
    tasks in the specified job do not reach Completed state within this time
    period, an exception will be raised.
    """

    timeout_expiration = datetime.datetime.now() + timeout
    print("Monitoring all tasks for 'Completed' state, timeout in {}..."
          .format(timeout), end='')

    while datetime.datetime.now() < timeout_expiration:
        print('.', end='')
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [task for task in tasks if
                            task.state != batchmodels.TaskState.completed]
        if not incomplete_tasks:
            print()
            return True
        else:
            time.sleep(1)

    print()
    raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within timeout period of " + str(timeout))

def wrap_commands_in_shell(ostype, commands):
    """Wrap commands in a shell

    :param list commands: list of commands to wrap
    :param str ostype: OS type, linux or windows
    :rtype: str
    :return: a shell wrapping commands
    """
    if ostype.lower() == 'linux':
        return '/bin/bash -c \'set -e; set -o pipefail; {}; wait\''.format(
            ';'.join(commands))
    elif ostype.lower() == 'windows':
        return 'cmd.exe /c "{}"'.format('&'.join(commands))
    else:
        raise ValueError('unknown ostype: {}'.format(ostype))

def select_latest_verified_vm_image_with_node_agent_sku(
        batch_client, publisher, offer, sku_starts_with):
    """Select the latest verified image that Azure Batch supports given
    a publisher, offer and sku (starts with filter).

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param str publisher: vm image publisher
    :param str offer: vm image offer
    :param str sku_starts_with: vm sku starts with filter
    :rtype: tuple
    :return: (node agent sku id to use, vm image ref to use)
    """
    # get verified vm image list and node agent sku ids from service
    node_agent_skus = batch_client.account.list_node_agent_skus()
    # pick the latest supported sku
    skus_to_use = [
        (sku, image_ref) for sku in node_agent_skus for image_ref in sorted(
            sku.verified_image_references, key=lambda item: item.sku)
        if image_ref.publisher.lower() == publisher.lower() and
        image_ref.offer.lower() == offer.lower() and
        image_ref.sku.startswith(sku_starts_with)
    ]
    # skus are listed in reverse order, pick first for latest
    sku_to_use, image_ref_to_use = skus_to_use[0]
    return (sku_to_use.id, image_ref_to_use)


def create_pool(batch_service_client, pool_id,
                resource_files, publisher, offer, sku,
                vm_size, node_count):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param list resource_files: A collection of resource files for the pool's
    start task.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sku
    """
    print('Creating pool [{}]...'.format(pool_id))

    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/

    # Specify the commands for the pool's start task. The start task is run
    # on each node as it joins the pool, and when it's rebooted or re-imaged.
    # If there were installs needed, this is where we'd add them as 
    task_commands = ['echo starting up']

    # Get the node agent SKU and image reference for the virtual machine
    # configuration.
    # For more information about the virtual machine configuration, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/
    sku_to_use, image_ref_to_use = \
        select_latest_verified_vm_image_with_node_agent_sku(
            batch_service_client, publisher, offer, sku)
    user = batchmodels.AutoUserSpecification(
        scope=batchmodels.AutoUserScope.pool,
        elevation_level=batchmodels.ElevationLevel.admin)
    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=image_ref_to_use,
            node_agent_sku_id=sku_to_use),
        vm_size=vm_size,
        target_dedicated_nodes=node_count,
        start_task=batch.models.StartTask(
            command_line=wrap_commands_in_shell('linux',task_commands),
            user_identity=batchmodels.UserIdentity(auto_user=user),
            wait_for_success=True,
            resource_files=resource_files),
    )

    try:
        batch_service_client.pool.add(new_pool)
    except batchmodels.batch_error.BatchErrorException as err:
        print(err)
        raise


def create_job(batch_service_client, job_id, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))

    job = batch.models.JobAddParameter(
        job_id,
        batch.models.PoolInformation(pool_id=pool_id))

    try:
        batch_service_client.job.add(job)
    except batchmodels.batch_error.BatchErrorException as err:
        print(err)
        raise




# Selecting files 
def create_resource_file_from_blob(block_blob_client, container_name, blob_name):
    """
    Creates a resource file object from a given blob client, container, and blob
    
    :param block_blob_client: A blob client with credentials.
    :type blob_blob_client: 'azure.storage.blob.BlockBlobService'
    :param str conatiner_name: The container name.
    :param str blob_name: The blob name.
    """
    sas_token = block_blob_client.generate_blob_shared_access_signature(
            container_name = container_name,
            blob_name = blob_name,
            permission=azureblob.BlobPermissions.READ,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        )

    sas_url = block_blob_client.make_blob_url(container_name,
                                            blob_name,
                                            sas_token=sas_token)

    return batchmodels.ResourceFile(file_path=blob_name,blob_source=sas_url)

def app_rename_rules_linux(input):
    """
    String rewrite rules for linux machines.

    :param str input: The string to be transformed to linux standard.
    """
    return input.replace(".","_").replace("#","_").replace("-","_")

def app_rename_rules(app, version=None, operating_system = "linux"):
    """
    Rewrites app and version strings to be consistent with the OS requirements.

    :param str app: The application name.
    :param str version: The version name.
    :param str operating_system: The OS, defaulting to linux.
    """
    output = ""
    if operating_system == "linux":
        output = "_{}".format(app_rename_rules_linux(app))
    else:
        output = "_{}".format(app)
    
    if version is not None and operating_system == "linux":
        output = output + "_{}".format(app_rename_rules_linux(version))
    if version is not None and operating_system != "linux":
        output = output + "#{}".format(version)

    return output

def query_yes_no(question, default="yes"):

    """

    Prompts the user for yes/no input, displaying the specified question text.



    :param str question: The text of the prompt for input.

    :param str default: The default if the user hits <ENTER>. Acceptable values

    are 'yes', 'no', and None.

    :rtype: str

    :return: 'yes' or 'no'

    """

    valid = {'y': 'yes', 'n': 'no'}

    if default is None:

        prompt = ' [y/n] '

    elif default == 'yes':

        prompt = ' [Y/n] '

    elif default == 'no':

        prompt = ' [y/N] '

    else:

        raise ValueError("Invalid default answer: '{}'".format(default))



    while 1:

        choice = input(question + prompt).lower()

        if default and not choice:

            return default

        try:

            return valid[choice[0]]

        except (KeyError, IndexError):

            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")





def print_batch_exception(batch_exception):

    """

    Prints the contents of the specified Batch exception.



    :param batch_exception:

    """

    print('-------------------------------------------')

    print('Exception encountered:')

    if (batch_exception.error and batch_exception.error.message and

            batch_exception.error.message.value):

        print(batch_exception.error.message.value)

        if batch_exception.error.values:

            print()

            for mesg in batch_exception.error.values:

                print('{}:\t{}'.format(mesg.key, mesg.value))

    print('-------------------------------------------')





def wrap_commands_in_shell(ostype, commands):

    """Wrap commands in a shell



    :param list commands: list of commands to wrap

    :param str ostype: OS type, linux or windows

    :rtype: str

    :return: a shell wrapping commands

    """

    if ostype.lower() == 'linux':

        return '/bin/bash -c \'set -e; set -o pipefail; {}; wait\''.format(

            ';'.join(commands))

    elif ostype.lower() == 'windows':

        return 'cmd.exe /c "{}"'.format('&'.join(commands))

    else:

        raise ValueError('unknown ostype: {}'.format(ostype))