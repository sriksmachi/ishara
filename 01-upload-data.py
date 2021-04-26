from azureml.core import Workspace, Dataset
ws = Workspace.from_config()

datastore = ws.get_default_datastore()

# Uncomment the below code when you want to upload the data to default data store

datastore.upload(src_dir='./Project_data',
                 target_path='datasets/gesture-data',
                 overwrite=True)

# create dataset
dataset = Dataset.File.from_files((datastore, 'datasets/gesture-data/**'))

# register dataset
gesture_ds = dataset.register(workspace=ws,
                                 name='gesture_ds',
                                 description='Dataset containing gesture images for 5 types of gestures')

print(gesture_ds)

