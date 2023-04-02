pip install gradio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import gradio as gr
import pickle
# Deploying the Multiple Polynomial regression model with degree 2,From the previous analysis it was the best model performing  on trqain data.
#Saving the model in pickle file 
poly_features = PolynomialFeatures(2)
x_poly_train = poly_features.fit_transform(np.array(df_abalone_train.loc[:,:"Shell weight"]))
regressor = LinearRegression()
regressor.fit(x_poly_train,df_abalone_train["Rings"])
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(regressor , open(filename, 'wb'))
# some time later...
filename = 'finalized_model.sav'
#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# Basic program implemented using gradio
# Basic program for deployng the above model
def predict(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight):
  l = [Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight]
  y_pred = loaded_model.predict(poly_features.fit_transform(np.array(l).reshape(1,-1)))
  return "The Predicted life of the abalone is "+str(round((y_pred[0]+1.5),2))+" Years"
max_values = dict(df_abalone.quantile(0.75))#Using quartile values as minimum and maximum to avoid out of bound predictions
min_values = dict(df_abalone.quantile(0.25))


# This can wrap any python function with specifying io as text,audio,video
demo = gr.Interface(fn = predict,inputs = [gr.Slider(min_values["Length"],max_values["Length"]),
                                           gr.Slider(min_values["Diameter"],max_values["Diameter"]),
                                           gr.Slider(min_values["Height"],max_values["Height"]),
                                           gr.Slider(min_values["Whole weight"],max_values["Whole weight"]),
                                           gr.Slider(min_values["Shucked weight"],max_values["Shucked weight"]),
                                           gr.Slider(min_values["Viscera weight"],max_values["Viscera weight"]),
                                           gr.Slider(min_values["Shell weight"],max_values["Shell weight"])],outputs = 'text',title="Predicting the life of Abalone based on its shell features")


demo.launch()
