from datetime import datetime as dt


date = dt.now().strftime("%Y%m%d")
time = dt.now().strftime("%H%M%S")

result_path = "../result/"
log_path = result_path + "logs/" + f"{date}/" + f"{time}"
model_path = result_path + "trained_models/"
csv_path = result_path + "csv/"
