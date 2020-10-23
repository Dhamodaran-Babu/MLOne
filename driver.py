import warnings
from validation import validate_data

warnings.filterwarnings("ignore")
data,status = validate_data('diabetes.csv')

if status :
    from preprocessing import preprocessor
    x,y,pbm_type = preprocessor(data)
    
    from splitter import train_test_validation_splitter
    xtrain,xval,ytrain,yval = train_test_validation_splitter(x,y,pbm_type)
    print("{} : {}".format(xtrain.shape,ytrain.shape))
    print("{} : {}".format(xval.shape,yval.shape))

    from classification_models import model_fitter
    all_estimators,results = model_fitter(xtrain,ytrain,xval,yval)

    print(all_estimators)
    print(results)
    
else :
    print("""Dataset didn't pass the criterions.Please make sure you have followed the guidelines properly.
    Data doesn't have sufficient rows and columns""")