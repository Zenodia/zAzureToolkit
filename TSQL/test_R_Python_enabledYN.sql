EXEC sp_execute_external_script  @language =N'Python',
@script=N'
print("Hello World")'

EXEC sp_execute_external_script  @language =N'R',
@script=N'
print("Hello World")'

EXEC sp_execute_external_script  @language =N'Python',
@script = N'
OutputDataSet = InputDataSet + 4;
',
@input_data_1 =N'SELECT 1 AS Col1';