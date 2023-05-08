from data_analysis import file_operate,preprocess,ploting,analysis
import pandas as pd
import numpy as np
import data
fop=file_operate.FileOperator()
prcesspor=preprocess.PreProcessor(fop)
fop.read_file('file1',r'data\sample',file_type='csv',sep=',',show_detail=False)
# fop.read_file('file2',r'data\SAR.csv',sep=',',show_detail=False)
fop.read_file('file3',r'data\detail.csv',sep=',',show_detail=False)
# fop.read_file('file4',r'data\附件1：data_100.csv',sep=',',show_detail=False)
# fop.read_file('file5',r'data\附件3：飞行参数测量数据.xlsx',sep=',',show_detail=False)

# d=fop.dfs['file1'].iloc[:,fop.get_cols('file1','int')]
# print(d)
# fop.get_details()
# fop.asdtype(dtypes={'file1':{'Length_(ft)':int,'Year':int,'single':int}})
# fop.get_details()

# prcesspor.drop_outliers()
# preprocess.drop_outliers(fop.dfs,fop.dfs_desc,cols={'file1': [3, 6, 7, 8, 9], 'file2': [2], 'file3': [8, 2, 4], 'file4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199], 'file5': [9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195]})

fop.get_details()
prcesspor.show_files_cols_desc()
fop.auto_dtype()
# prcesspor.show_files_cols_desc()
prcesspor.fillna()
# prcesspor.dropna()
prcesspor.drop_outliers(draw=False)
# prcesspor.norm()
# prcesspor.standard()
# prcesspor.show_files_cols_desc()
fop.get_details()
# prcesspor.show_files_cols_desc()
# prcesspor.fillna(fids=['file1'],float_miss=np.nan,str_miss="")
# prcesspor.fillna(fids=['file2'],float_strategy="mean",str_strategy="constant",const="-1",float_miss=np.nan,str_miss="")
# preprocess.fillna(dfs=fop.dfs,fids=['file3'],)





ayzr=analysis.Analyzer(fop)
d1=ayzr.corr_matrix('file1',[3, 6, 7, 8, 9])
ploting.corr_heatmap(d1.values,d1.columns)
sd=ayzr.corr_unique_values('file1')
ploting.scatter(sd['x'],sd['y'],show=True)
ploting.linear(sd['x'],sd['y'],show=True)
d2=ayzr.corr_unique_group('file1',value='min',by=['Year'],cols=['single','gdp','Length_(ft)'])
ploting.bar(d2['values'],group_names=d2['by'].astype(str),show=True,names=d2['col_names'][1:])

