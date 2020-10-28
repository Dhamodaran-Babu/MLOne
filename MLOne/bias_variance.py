from mlxtend.evaluate import bias_variance_decomp

def bv_decomp(all_estimators, X_train, y_train, X_test, y_test):
    print("\n\n<<<<DECOMPOSING THEBIAS AND VARIANCE>>>>\n\n")
    for key,value in all_estimators.items():
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
                value, X_train, y_train, X_test, y_test, 
                loss='0-1_loss',
                random_seed=123)
        print('Average expected loss for {} is {}'.format(key,round(avg_expected_loss,3)))
        print('Average bias for {} is {}'.format(key,round(avg_bias,3)))
        print('Average variance for {} is {}'.format(key,round(avg_var,3)))
        print('\n')

