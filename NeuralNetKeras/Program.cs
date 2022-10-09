using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;

using Accord.Math;
using MNIST.IO;
using System.Diagnostics;
using Accord.IO;
using System.IO;
using System.Xml.Serialization;
using System.Xml;
using Python.Runtime;
using System.Diagnostics.Tracing;

//Load train data
var trainData = FileReaderMNIST.LoadImagesAndLables(
"./data/train-labels-idx1-ubyte.gz",
"./data/train-images-idx3-ubyte.gz");

var testData = FileReaderMNIST.LoadImagesAndLables(
    "./data/t10k-labels-idx1-ubyte.gz",
    "./data/t10k-images-idx3-ubyte.gz");


//Build sequential model

var layers = new List<ILayer>();
layers.Add(keras.layers.Dense(64, keras.activations.Relu, input_shape: new Shape(28*28)));
for (int i=0; i<5;i++) layers.Add(keras.layers.Dense(64, keras.activations.Relu));
layers.Add(keras.layers.Dense(10, keras.activations.Softmax));

var model = keras.Sequential(layers: layers);
for (int i = 1, c = layers.Count; i < c; i++) model.Layers.Add(layers[i]);

// Model Compile and Summary
model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "accuracy" });
model.summary();

List<float[]> x_list = new();
List<float[]> y_list = new();

foreach (var image in trainData)
{
  x_list.Add(byteArrToFloat(image.Image));
  y_list.Add(byteToFloatAnswer(image.Label));
}
var x_train2 = CreateRectangularArray(x_list);
var x_train = np.array(x_train2);
var y_train2 = CreateRectangularArray(y_list);
var y_train = np.array(y_train2);


Stopwatch sw = new();
sw.Start();
// Model Train
model.fit(x_train, y_train, batch_size: 1000, epochs: 10, verbose: 1, use_multiprocessing: true);
sw.Stop();

Console.WriteLine($"Done! Total train time: {sw.Elapsed}");

////Save model and weights
//string json = model.ToJson();
//File.WriteAllText("model.json", json);
//model.SaveWeight("model.h5");

////Load model and weight
//var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
//loaded_model.LoadWeight("model.h5");


static float[] byteArrToFloat(byte[,] byteArr)
{
  int indice = 0;
  float[] imagem = new float[784];
  for (int i = 0; i < 28; i++)
  {
    for (int j = 0; j < 28; j++)
    {
      double d = byteArr[i, j] / 255.0;

      imagem[indice] = Convert.ToSingle(d);
      indice++;
    }
  }
  return imagem;
}

static float[] byteToFloatAnswer(byte ans)
{
  switch (ans)
  {
    case 0:
      return new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    case 1:
      return new float[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
    case 2:
      return new float[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    case 3:
      return new float[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    case 4:
      return new float[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
    case 5:
      return new float[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    case 6:
      return new float[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
    case 7:
      return new float[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
    case 8:
      return new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
    case 9:
      return new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    default:
      return new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  }
}

static T[,] CreateRectangularArray<T>(IList<T[]> arrays)
{
  // TODO: Validation and special-casing for arrays.Count == 0
  int minorLength = arrays[0].Length;
  T[,] ret = new T[arrays.Count, minorLength];
  for (int i = 0; i < arrays.Count; i++)
  {
    var array = arrays[i];
    if (array.Length != minorLength)
    {
      throw new ArgumentException
          ("All arrays must be the same length");
    }
    for (int j = 0; j < minorLength; j++)
    {
      ret[i, j] = array[j];
    }
  }
  return ret;
}