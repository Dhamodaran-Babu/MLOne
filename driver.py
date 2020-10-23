from validation import validate_data

data,status = validate_data('diabetes.csv')

if status :
    from preprocessing import preprocessor
    x,y = preprocessor(data)
    print("{} : {}".format(x,y))

else :
    print("Data doesn't have sufficient rows and columns")