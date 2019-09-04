package machinelearning;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisClassifier {

  private static final int CLASSES_COUNT = 3;
  private static final int FEATURES_COUNT = 4;

  public static void main(String[] args) throws IOException, InterruptedException {

    Map<Integer, String> flowerMap = new HashMap<>();
    flowerMap.put(0, "Iris-setosa");
    flowerMap.put(1, "Iris-versicolor");
    flowerMap.put(2, "Iris-virginica");

    DataSet allData;
    try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
      recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

      DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
      allData = iterator.next();
    }

    DataSet allDataTest;
    try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
      recordReader.initialize(new FileSplit(new ClassPathResource("iris-test.txt").getFile()));

      DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 30, FEATURES_COUNT, CLASSES_COUNT);
      allDataTest = iterator.next();
    }

    allData.shuffle();

    // Neural nets all about numbers. Lets normalize our data
    DataNormalization normalizer = new NormalizerStandardize();
    // Collect the statistics from the training data. This does not modify the input data
    normalizer.fit(allData);
    normalizer.transform(allData);
    normalizer.transform(allDataTest);

    MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        .iterations(1000)
        .activation(Activation.TANH)
        .weightInit(WeightInit.XAVIER)
        .learningRate(0.1)
        .regularization(true).l2(0.0001)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3)
            .build())
        .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
            .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(3).nOut(CLASSES_COUNT).build())
        .backprop(true).pretrain(false)
        .build();

    SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(120);
    DataSet trainingData = testAndTrain.getTrain();
//    DataSet testData = testAndTrain.getTest();
//    DataSet trainingData = allData;
    DataSet testData = allDataTest;

    MultiLayerNetwork model = new MultiLayerNetwork(configuration);
    model.init();
    model.fit(trainingData);

    INDArray output = model.output(testData.getFeatureMatrix());

    Evaluation eval = new Evaluation(CLASSES_COUNT);
    eval.eval(testData.getLabels(), output);
    System.out.println(eval.stats());
    classify(output, flowerMap);
  }

  private static void classify(INDArray output, Map<Integer, String> flowerMap) {
    for (int i = 0; i < output.rows(); i++) {
      final int max = maxIndex(getFloatArrayFromSlice(output.slice(i)));
      System.out.println(Integer.toString(i+1) + "-" + output.slice(i)+ " => " + max + " => " + flowerMap.get(max));
    }
  }

  private static float[] getFloatArrayFromSlice(INDArray rowSlice) {
    float[] result = new float[rowSlice.columns()];
    for (int i = 0; i < rowSlice.columns(); i++) {
      result[i] = rowSlice.getFloat(i);
    }
    return result;
  }

  private static int maxIndex(float[] vals) {
    int maxIndex = 0;
    for (int i = 1; i < vals.length; i++) {
      float newnumber = vals[i];
      if ((newnumber > vals[maxIndex])) {
        maxIndex = i;
      }
    }
    return maxIndex;
  }

}
