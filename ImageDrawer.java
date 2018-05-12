
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.image.*;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import java.util.Random;

public class ImageDrawer extends Application {

    private Image originalImage; //On affiche l'image original sur la gauche de la fenêtre
    private WritableImage composition; //Destination de la génération
    private MultiLayerNetwork nn;
    private INDArray xyOut;
    private final Random r = new Random();

    private void onCalc(){
        int batchSize = 1000;
        int numBatches = 5;
        for (int i =0; i< numBatches; i++){
            DataSet ds = generateDataSet(batchSize);
            nn.fit(ds);
        }
        drawImage();
        Platform.runLater(this::onCalc);
    }

    @Override
    public void init(){
        originalImage = new Image("/DataExamples/goku.png");

        final int w = (int) originalImage.getWidth();
        final int h = (int) originalImage.getHeight();
        composition = new WritableImage(w, h); //Right image.

        nn = createNN();

        boolean fUseUI = false; 
        if(fUseUI) {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            nn.setListeners(new StatsListener(statsStorage));
        }

        int numPoints = h * w;
        xyOut = Nd4j.zeros(numPoints, 2);
        for (int i = 0; i < w; i++) {
            double xp = scaleXY(i,w);
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                double yp = scaleXY(j,h);

                xyOut.put(index, 0, xp); //2 inputs. x and y.
                xyOut.put(index, 1, yp);
            }
        }
        drawImage();
    }

    @Override
    public void start(Stage primaryStage) {

        final int w = (int) originalImage.getWidth();
        final int h = (int) originalImage.getHeight();
        final int zoom = 1; 

        ImageView iv1 = new ImageView(); 
        iv1.setImage(originalImage);
        iv1.setFitHeight( zoom* h);
        iv1.setFitWidth(zoom*w);

        ImageView iv2 = new ImageView();
        iv2.setImage(composition);
        iv2.setFitHeight( zoom* h);
        iv2.setFitWidth(zoom*w);

        HBox root = new HBox(); //build the scene.
        Scene scene = new Scene(root);
        root.getChildren().addAll(iv1, iv2);

        primaryStage.setTitle("Neural Network Drawing Demo.");
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(we -> System.exit(0));
        primaryStage.show();

        Platform.setImplicitExit(true);
        Platform.runLater(this::onCalc);
    }

    public static void main( String[] args )
    {
        launch(args);
    }

    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        double learningRate = 0.05;
        int numInputs = 2;   // x and y.
        int numHiddenNodes = 100;
        int numOutputs = 3 ; //R, G and B value.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learningRate, 0.9))
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                .activation(Activation.IDENTITY)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    private  DataSet generateDataSet(int batchSize) {
        int w = (int) originalImage.getWidth();
        int h = (int) originalImage.getHeight();

        PixelReader reader = originalImage.getPixelReader();

        INDArray xy = Nd4j.zeros(batchSize, 2);
        INDArray out = Nd4j.zeros(batchSize, 3);

        for (int index = 0; index < batchSize; index++) {
            int i = r.nextInt(w);
            int j = r.nextInt(h);
            double xp = scaleXY(i,w);
            double yp = scaleXY(j,h);
            Color c = reader.getColor(i, j);

            xy.put(index, 0, xp); //2 inputs. x and y.
            xy.put(index, 1, yp);

            out.put(index, 0, c.getRed());  //3 outputs. the RGB values.
            out.put(index, 1, c.getGreen());
            out.put(index, 2, c.getBlue());
        }
        return new DataSet(xy, out);
    }

    private void drawImage() {
        int w = (int) composition.getWidth();
        int h = (int) composition.getHeight();

        INDArray out = nn.output(xyOut);
        PixelWriter writer = composition.getPixelWriter();

        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                double red = capNNOutput(out.getDouble(index, 0));
                double green = capNNOutput(out.getDouble(index, 1));
                double blue = capNNOutput(out.getDouble(index, 2));

                Color c = new Color(red, green, blue, 1.0);
                writer.setColor(i, j, c);
            }
        }
    }

    private static double capNNOutput(double x) {
        double tmp = (x<0.0) ? 0.0 : x;
        return (tmp > 1.0) ? 1.0 : tmp;
    }

    private static double scaleXY(int i, int maxI){
        return (double) i / (double) (maxI - 1) -0.5;
    }
}
