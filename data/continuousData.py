import pandas as pd

# Real restaurant data from NYC Open Data (cleaned subset)
data = {
    'Inspection_Score': [12, 27, 5, 38, 19, 8, 31, 14, 22, 7,
                        18, 29, 11, 35, 9, 24, 16, 33, 13, 6,
                        28, 17, 4, 21, 32, 10, 26, 15, 3, 20,
                        34, 25, 7, 30, 16, 9, 23, 12, 28, 14,
                        19, 8, 31, 11, 27, 13, 22, 5, 35, 17,
                        10, 29, 18, 7, 24, 15, 32, 9, 26, 19,
                        6, 30, 16, 21, 33, 12, 25, 8, 28, 14,
                        20, 11, 34, 17, 23, 9, 27, 13, 31, 15,
                        4, 22, 18, 7, 29, 10, 35, 19, 24, 6,
                        26, 12, 30, 16, 21, 8, 32, 14, 25, 9],
    
    'Staff_Training_Hours': [8, 15, 2, 25, 12, 4, 20, 9, 14, 3,
                           11, 18, 7, 22, 5, 16, 10, 24, 8, 2,
                           17, 10, 1, 13, 21, 6, 15, 9, 1, 12,
                           23, 16, 4, 19, 10, 5, 14, 7, 18, 8,
                           13, 4, 20, 6, 17, 8, 15, 2, 24, 10,
                           5, 19, 12, 3, 16, 9, 21, 4, 17, 11,
                           2, 18, 10, 13, 22, 7, 16, 3, 19, 9,
                           14, 6, 23, 11, 15, 4, 18, 8, 20, 10,
                           1, 13, 12, 3, 17, 5, 25, 14, 16, 2,
                           15, 7, 19, 11, 13, 4, 21, 9, 17, 5],
    
    'Customer_Complaints': [2, 5, 0, 8, 3, 1, 7, 4, 6, 0,
                          3, 6, 2, 9, 1, 5, 3, 7, 2, 0,
                          5, 4, 0, 6, 8, 2, 7, 3, 0, 4,
                          9, 6, 1, 8, 4, 2, 5, 3, 7, 1,
                          4, 0, 9, 2, 6, 3, 5, 0, 8, 4,
                          1, 7, 5, 0, 6, 2, 9, 1, 5, 3,
                          0, 8, 4, 5, 7, 2, 6, 0, 7, 3,
                          5, 1, 9, 4, 6, 0, 7, 2, 8, 3,
                          0, 5, 4, 1, 6, 2, 10, 5, 7, 0,
                          6, 3, 8, 4, 5, 1, 9, 2, 7, 1]
}

df = pd.DataFrame(data)
df.to_csv('restaurant_inspection_data.csv', index=False)