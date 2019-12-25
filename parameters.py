from datetime import date, timedelta

num_epochs = 6
learning_rate = 0.006
output_period = 10
test_output_period = 1
batch_size = 50
augmented_training_data = True
path = "models/model.%s" % date.today()+"_"+str(learning_rate)