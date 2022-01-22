from library import *

IMG_SIZE = (224,224,3)
EPOCH = 100
BATCH_SIZE = 32

def fine_tune(model,n):
  model.trainable = True
  if n != 0:
    for layer in model.layers[:n]:
      layer.trainable =  False
  return model
  
def details_layers(model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

def read_data(csv_data,img_file,imge_resize=IMG_SIZE):
  # print(csv_data.head())
  # print(csv_data[['Id','Pawpularity']])
  # print(csv_data.isnull().sum())
  # print(csv_data.index)

  imgs = []
  labels = []
  for i in tqdm.tqdm(csv_data.index):
      img_id = csv_data.loc[i,'Id']
      label_id = csv_data.loc[i,'Pawpularity']

      img = cv.imread(img_file+"\\"+img_id+".jpg")
      img = cv.resize(img, (imge_resize[0], imge_resize[1]))
      imgs.append(img)
      labels.append(label_id)
  img = np.array(imgs)/255.0
  labels = np.array(labels).reshape(-1,1)
  return img,labels

path = "D:\\Data\\animal"
csv_file = os.path.join(path,'train.csv')
img_file = os.path.join(path, 'train')
csv_data = pd.read_csv(csv_file, index_col=False)


img,labels = read_data(csv_data,img_file)
# normlization data
# mX = np.max(labels)
# mI = np.min(labels)
# labels = (labels - mI)/(mX-mI)

# print(labels.reshape(-1,1).shape)
sc = StandardScaler()
labels = sc.fit_transform(labels)
print(labels.shape)

train_x, test_x, train_y, test_y = train_test_split(img, labels, test_size=0.2, random_state=52)

print(f"train dim: {train_x.shape} , test dim: {test_x.shape} , train_y:{train_y.shape}")
print(train_y)
fig = plt.figure(figsize=(12,12))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(img[i])
    plt.title(labels[i])
    plt.tight_layout(h_pad=2,w_pad=2)

plt.show() 

callback = [
    EarlyStopping(monitor='val_loss',
                patience=5,
                verbose=0),
    ReduceLROnPlateau(monitor='val_loss',
                    patience=3,
                    verbose=1)
]

model = VGG16(input_shape=IMG_SIZE, include_top=False)

x = GlobalAveragePooling2D()(model.output)
x = Dense(256, activation="relu")(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64,activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(32,activation="relu")(x)
prediction = Dense(1,activation='linear')(x)

model = Model(inputs=model.input , outputs=prediction)

model = fine_tune(model,19)
details_layers(model)

model.compile(optimizer=Adam(learning_rate=0.0001) , loss="mean_squared_error", metrics=['accuracy'])

print(model.summary())


history = model.fit(train_x, train_y,
                    validation_data = (test_x,test_y),
                    verbose = 1,
                    epochs = EPOCH,
                    batch_size= BATCH_SIZE,
                    callbacks=[callback])

path = "Pawpularity_net.h5"
history_result_path = "history.csv"
model.save(path)

pf = pd.DataFrame(history.history)
with open(history_result_path,mode='w') as f:
  pf.to_csv(f)





