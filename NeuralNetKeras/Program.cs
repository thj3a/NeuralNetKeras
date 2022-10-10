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
using Accord.DataSets;
using System.IO;
using System.Xml.Serialization;
using System.Xml;
using Python.Runtime;
using System.Diagnostics.Tracing;
using Microsoft.VisualBasic.FileIO;
using Keras.Datasets;
using Tensorflow.Keras.Datasets;
using System.Drawing;

//Load train data
var trainData = FileReaderMNIST.LoadImagesAndLables(
@".\..\..\..\data\train-labels-idx1-ubyte.gz",
@".\..\..\..\data\train-images-idx3-ubyte.gz");
// Load test data
var testData = FileReaderMNIST.LoadImagesAndLables(
    @".\..\..\..\data\t10k-labels-idx1-ubyte.gz",
    @".\..\..\..\data\t10k-images-idx3-ubyte.gz");

Sequential model = CreateModel(5, 64, 28 * 28, 10);
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
model.fit(x_train, y_train, batch_size: 1000, epochs: 5, verbose: 1, workers: 30);
sw.Stop();

Console.WriteLine($"Done! Total train time: {sw.Elapsed}");

//// Save model
//model.save(SpecialDirectories.MyDocuments+ "\\model.h5");
//model.save_weights(SpecialDirectories.MyDocuments + "\\model1.h5");
//// nullify the model
//model = null;

//// Reinstantiate the model 
////model = CreateModel(5, 64, 28 * 28, 10);
//// Load weights
//model = keras.Sequential();
//model.load_weights(SpecialDirectories.MyDocuments + "\\model1.h5");

//Console.WriteLine("Done");


static Sequential CreateModel(int layers, int neurons, int input_size, int output_size)
{
  // Prepare layers
  var list_layers = new List<ILayer>();
  list_layers.Add(keras.layers.Dense(neurons, keras.activations.Relu, input_shape: new Shape(input_size)));
  for (int i = 0; i < layers; i++) list_layers.Add(keras.layers.Dense(64, keras.activations.Relu));
  list_layers.Add(keras.layers.Dense(output_size, keras.activations.Softmax));

  //Build sequential model
  var model = keras.Sequential(layers: list_layers);
  for (int i = 1, c = list_layers.Count; i < c; i++) model.Layers.Add(list_layers[i]);

  // Model Compile
  model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "accuracy" });
  
  return model;
}

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
