a pipeline for quick data analysis and machine learning

file_operate:
    read file and do base operate like auto asdtype etc.
    it support multi type files operates together.
preprocess:
    create a preprocessor to preprocessing(dropna,fillna,dropoutlyers...)
    you can select the file and cols by passing dict params.
analysis:
    base on the previous manipulations,we get clean datas,we can now start the real analytical work:
    overview:correlation heatmap
    single or multi cols: line,scatter,bar,
    group cols:line,scatter,bar,pie
modeling:
    we provide base ml models to complete classification or regression tasks
    listing:
        gbdt:xgboost,light gbm,radom forests
        norm:svc,linear,logistic,bayes
        timesequence:ARMA,ARIMA
        nn:
        DeepLearning:\

statistic_test:https://blog.csdn.net/weixin_46271668/article/details/123981051
    normality test
    correlation test
    significance test
    parametric test
    nonparametric test