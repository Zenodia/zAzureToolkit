import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import azure.storage.blob as azureblob


if __name__ == "__main__":
    # Example Syntax
    # python .\batch\make_lm.py -i .\data\split\splitaustralia.csv --storageaccount azueus2devsabatch --storagecontainer output --key EBVkUou5ePWV0qvGqQtrCHgzuk45ltfdExylQQLkGGFYUgXaERZJUyY2UGq2opwgs4wj9XuJGtKVZlY6BhtN8Q==
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', help = "The location of the input file", required = True)
    """
    parser.add_argument('--storageaccount', required=True,
                        help='The name the Azure Storage account that owns the blob storage container to which to upload results.')
    parser.add_argument('--storagecontainer', required=True,
                        help='The Azure Blob storage container')
    parser.add_argument('--key', required=True,
                        help='The access key providing write access to the Storage container.')
    """
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile,sep=',')
    #df=pd.read_csv('../example_data/breast_cancer.csv')
    
    features = list(df.columns[:-1])
    
    X = df.loc[:,features]
    Y = df.loc[:,["los"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123456)
    # import all models
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    models=[]
    models.append(('GLM',LogisticRegression()))
    models.append(('KNN',KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
    models.append(('TREE',DecisionTreeClassifier(criterion = 'entropy', random_state = 0)))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)))

    acc=0
    best_model=None
    best_model_name=None
    for name, model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        #cm = confusion_matrix(y_test, y_pred)
        print ("---------- model: " + name + '   performance as following --------->')
        print("Acurracy: " + str(round(accuracy_score(y_test, y_pred),4)))

        if acc<round(accuracy_score(y_test, y_pred),4):
            acc=round(accuracy_score(y_test, y_pred),4)
            best_model=model
            best_model_name=name
 
    
    
    predicted=best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print("accuracy :",accuracy)
    dtest=pd.DataFrame(X_test,columns=features)
    dtest.head()
    dtest['predict']=predicted
    dtest['TrueCancer']=y_test
    print("predicted data sample")
    print(dtest.head())
    
    
    # save the model to disk
    filename = '{}_{}_model.pkl'.format(os.path.splitext(args.inputfile)[0],best_model_name)
    print("filename",filename)
    pickle.dump(best_model, open(filename, 'wb'))
    
     
    # some time later...
     
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)
    
    #output_file_path = os.path.realpath(output_file)
    model_output_path=os.path.realpath(filename)
    
    #dtest.to_csv('{}_pred.csv'.format(os.path.splitext(args.inputfile)[0]), index=False)

    #print("output files complete !")
    

    output_model_path = os.path.realpath(model_output_path)
    #output_file_path= os.path.realpath('{}_pred.csv'.format(os.path.splitext(args.inputfile)[0]))
    print("output model path",output_model_path)
    #print("output pred file path", output_file_path)
    """
    blob_client = azureblob.BlockBlobService(account_name=args.storageaccount, account_key=args.key)

    blob_client.create_blob_from_path(args.storagecontainer,filename,output_model_path)
    """
    #blob_client.create_blob_from_path(args.storagecontainer,'{}_pred.csv'.format(os.path.splitext(args.inputfile)[0]),output_file_path)