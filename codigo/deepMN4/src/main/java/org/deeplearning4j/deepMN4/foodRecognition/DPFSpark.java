package org.deeplearning4j.deepMN4.foodRecognition;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.api.IterationListener;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.data.DataSetExportFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.deepMN4.TestModels.AlexNet;
import org.deeplearning4j.eval.Evaluation;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;



import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Random;



public class DPFSpark{
//Esto esta basado en el proyecto ComputerVision del paquete ejemplos de github.
//Por lo tanto se necesitara usar el modulo TestModels


    protected static final Logger log = LoggerFactory.getLogger(DPFSpark.class);

    protected static final int height = 64; // original is 250
    protected static final int width = 64;
    protected static final int channels = 3;
    protected static final int numLabels = 14; // LFWLoader.NUM_LABELS;
    protected static final int numSamples = 100; //LFWLoader.SUB_NUM_IMAGES - 4;
    protected static int batchSize = 32;
    protected static int iterations = 35;
    protected static int seed = 123;
    protected static boolean useSubset = false;
    protected static double splitTrainTest = 0.8;

    public static void main(String[] args) throws Exception {

        int listenerFreq = batchSize;
        int epochs = 1;
        Random rng = new Random(seed);
        int contador;
// Load Data.

        log.info("Loading the data....");
        System.out.println("Loading the data");
        File mainPath = new File(System.getProperty("user.dir"), "data/");
        System.out.println(mainPath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng,NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        DataSetIterator dataIter;
        List<DataSet> data;
  

// split the data in train and test.
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 80,  20);
        InputSplit  trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];
        // Llegado a este punto he dividido la data
        System.out.println("He usado el Input Split object");


        //Extendiendo el dataset.

        //Segun mi teoria aqui uso 3 transformaciones es decir el dataset original lo extiendo 3 veces. Las dos primeras.
        //
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random (123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
  	    // Esta funcion no esta implementada en la version 6.0//  ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {flipTransform1, warpTransform, flipTransform2});




        //Setup SparkContext with critical parameters
        log.info("Setup spark enviroment...."); 
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DPF");
        sparkConf.set("spark.driver.maxResultSize", "5G");
        sparkConf.set("spark.executor.heartbeatInterval","10000000");
        sparkConf.set("spark.network.timeout","10000000");
        
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
       //

        log.info("Load the AlexNet Model");
        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();
        network.init();
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Setup parameter averaging
        //Configuration for Spark training
       // ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
         TrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(2)
                .saveUpdater(true)
                .averagingFrequency(10)
                .batchSizePerWorker(batchSize)
                .repartionData(Repartition.Always)
                .repartitionStrategy(RepartitionStrategy.SparkDefault)
                .build();

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, network, trainMaster);
  //      sparkNetwork.setCollectTrainingStats(true);
		JavaRDD<DataSet> sparkDataTrain;    
		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
    int t=1;
        for(ImageTransform transform: transforms) {
        	contador=1;
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            data = new ArrayList<>();
        	while(dataIter.hasNext()){
            	System.out.println("subiendo los datos#:"+contador);
            	data.add(dataIter.next());
            	contador++;
            }
         	sparkDataTrain = sc.parallelize(data);
	        log.info("Train model...");
	        for( int i=0;i<epochs;i++)
	        {
	            sparkNetwork.fit(sparkDataTrain);
	           // sparkDataTrain.deleteTempFiles();
	            trainMaster.deleteTempFiles(sc);
	            System.out.println("******************************************************************************");
	            System.out.println("******************************************************************************");
	            System.out.println("Complete epoch " +i + "transformacion numero:"+t);
	            System.out.println("******************************************************************************");
	            System.out.println("******************************************************************************");
	         
	        }

	        t++; 
        }

		

    //    SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
    //  	StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc);

        contador=1;
        ImageRecordReader recordReader_test = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader_test.initialize(testData, null);
        DataSetIterator dataIter_test;
        dataIter_test = new RecordReaderDataSetIterator(recordReader_test, batchSize, 1, numLabels);
        List<DataSet> data_test = new ArrayList<>();
        while(dataIter_test.hasNext()){
            System.out.println("subiendo los datos de test #:"+contador);
            data_test.add(dataIter_test.next());
            contador++;
        }

        JavaRDD<DataSet> testData1 = sc.parallelize(data_test);

        Evaluation evaluation = sparkNetwork.evaluate(testData1);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

     //   SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
      //  StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc);
      //  System.out.println("----- DONE -----");
      //  log.info("****************Example finished********************");


    }
}
