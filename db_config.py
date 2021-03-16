# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:14:21 2021

@author: knum
"""
import pandas as pd
import pyodbc
import MySQLdb

server = 'dementiaapi.database.windows.net'
database = 'dementiadata'
username = 'dementiaapi'
password = 'm,./12369'   
driver= '{ODBC Driver 17 for SQL Server}'

def send_package(filename, write, delete):
    data = pd.read_csv (r'./'+filename)   
    df = pd.DataFrame(data, columns= ['frame','part_score','x_coord','y_coord'])
    
    with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
        with conn.cursor() as cursor:
    
            ''' Create Table '''
            # cursor.execute('CREATE TABLE motion_data (Frame int, score float, x_coord float, y_coord float)')
            
            '''Write Content'''
            if write:
                for row in df.itertuples():
                    cursor.execute('''
                                INSERT INTO dementiadata.dbo.motion_data (Frame, score, x_coord, y_coord)
                                VALUES (?,?,?,?)
                                ''',
                                row.frame, 
                                row.part_score,
                                row.x_coord,
                                row.y_coord
                                )
            
            '''Delete Content'''
            if delete:
                # sql = "DROP TABLE motion_data"
                sql = "DELETE FROM motion_data"
                cursor.execute(sql)
            
            conn.commit()
            
            # print("Contents of the table after delete operation ")
            # cursor.execute("SELECT * from motion_data")
            # print(cursor.fetchall())

# send_package('coba_gui.csv')