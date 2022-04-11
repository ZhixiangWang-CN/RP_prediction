### RP_prediction
***
This project is comes from the article <br>
" Computed tomography and radiation dose images-based deep-learning model for predicting radiation pneumonitis in lung cancer patients after radiation therapy: A pilot study with external validation"
***
The online project was shown in:

https://flask-web-zx.herokuapp.com/

The demo data can be downloaded from the folder DemoData
***




For training the model in your own data:
1. create your dataset.json by Tools/convert_dataset_to_json
2. Change the Parameters/Img_classification.json according your dataset and your model
3. run 
```python Training.py```

For test your model:
run
```python Test.py```


For Fine tuining the model by other dataset:
run
```python Fine_tuining.py```

For plot the attention map:
run
```python Test_and_attention_plot3D```
