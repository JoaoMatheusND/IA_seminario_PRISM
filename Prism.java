//Baixar o weka via https://weka.softonic.com.br/#:~:text=Weka%20é%20um%20aplicativo%20de,esquemas%20de%20aprendizado%20de%20máquina.
//Código gerado via ChatGPT

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Prism;

public class PRISMExample {
    public static void main(String[] args) throws Exception {
        
        // Carregando conjunto de dados
        DataSource source = new DataSource("path/to/your/dataset.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Criando e configurando o classificador PRISM
        Prism prism = new Prism();
        prism.setUseDefaultRules(true); // Usar regras padrão ou definir suas próprias regras

        // Avaliando o classificador usando validação cruzada de 10 folds
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(prism, data, 10, new java.util.Random(1));

        // Imprimindo resultados
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
