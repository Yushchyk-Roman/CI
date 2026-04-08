from data_preparetion import preparetion
from config import *

def menu_info():
    print(f"{INDENT} MENU: ")
    print("")

def menu():    
    data_report = input("Preparing data, do you need data report? [y, N]: ") 
    data_report = True if data_report == "y" else False
    
    x_train, x_test, y_train, y_test = preparetion(PATH_TO_TRAIN_CSV, PATH_TO_TEST_CSV, need_report=data_report)
    
    user_choice = ''
    while user_choice not in ["exit", "вихід", 7]:
        try:
            menu_info()
            
            user_choice = int(input("Виберіть опцію: "))
            match user_choice:
                case 1:
                    print("lol")
                case 7:
                    print("Exit...")
                case _:
                    print("Не правильний ввід")
            
            
        
        
        except  Exception as e:
            print(e)
            
        