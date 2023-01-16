def Prediction(model,x_test):

    # predict
    predictions = model.predict(x_test)
    predictions = predictions > 0.5 # for binary

    print(predictions)
    
    #return predictions
