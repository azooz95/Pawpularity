from library import *

path = "D:\\Data\\animal"
csv_file = os.path.join(path,'train.csv')
img_file = os.path.join(path, 'train')
csv_data = pd.read_csv(csv_file, index_col=False)
IMG_SIZE = (224,224)
def read_data(csv_data,img_file,imge_resize=IMG_SIZE):
  # print(csv_data.head())
  # print(csv_data[['Id','Pawpularity']])
  # print(csv_data.isnull().sum())
  # print(csv_data.index)

  imgs = []
  ids = []
  for i in tqdm.tqdm(csv_data.index):
      img_id = csv_data.loc[i,'Id']
      label_id = csv_data.loc[i,'Pawpularity']

      img = cv.imread(img_file+"\\"+img_id+".jpg")
      img = cv.resize(img, (imge_resize[0], imge_resize[1]))
      imgs.append(img)
      ids.append(img_id)
  img = np.array(imgs)/255.0
  return img,ids


img, ids = read_data(csv_data,img_file)

model_path = "Pawpularity_net.h5"

model = tf.keras.models.load_model(model_path)
model = Model(inputs=model.input , outputs= model.layers[-2].output)
pred = model.predict(img,verbose=1,batch_size=32)
columns = range(csv_data.shape[1],pred.shape[1]+csv_data.shape[1])
print(len(columns))
print(list(columns))
print(pred.shape)
print(csv_data.shape)
csv_data.loc[:,list(columns)] =pred
print(csv_data.head())


file_saved = 'new_file.csv'
csv_data.to_csv(file_saved, index=False)

