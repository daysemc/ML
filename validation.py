def Evaluate(model,history,x_evaluate,y_evaluate):

    # print valores de loss e accuracy para o dataset 'test
    val_loss,val_acc = model.evaluate(x_evaluate,y_evaluate)
    print(val_loss,val_acc)

    # plotar graficos de loss e accuracy para os datasets 'train' e 'test'
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    #plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()

    plt.show()
    
