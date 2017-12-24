using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Task3
{
    public partial class Form2 : Form
    {
        NeuralNetwork neuralNetwork;

        // Картинки + метки
        List<ReadMnist.DigitImage> mnistInfo;
        List<ReadMnist.DigitImage> mnistTestInfo;

        // Кол-во скрытых слоев
        int cntLayers;
        // Кол-во нейронов на скрытых слоях
        int[] cntNeurons;

        public Form2(int cntLayers, int[] cntNeurons)
        {
            InitializeComponent();
            this.cntLayers = cntLayers;
            
            // Если не по умолчанию, то заполняем
            if (cntLayers >= 1 && !cntNeurons[0].Equals(0))
                this.cntNeurons = cntNeurons;
        }
        

        // Получение данных для сети
        private void learnNetworkToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                ReadMnist.ReadMnistData(out mnistInfo);
                ReadMnist.ReadMnistData(out mnistTestInfo, 0);
                cntNeurons = new int[cntLayers];
                cntNeurons[0] = mnistInfo.Count;
                neuralNetwork = new NeuralNetwork(mnistInfo, cntLayers, cntNeurons, mnistTestInfo);
                MessageBox.Show("Данные получены");
            }
            catch (System.OutOfMemoryException)
            {
                MessageBox.Show("Данные не получены");
            }
        }

        private static void AddAnswers(ref List<List<double>> answers)
        {
            FolderBrowserDialog fileDialog = new FolderBrowserDialog();

            if (fileDialog.ShowDialog() == DialogResult.OK)
            {
                var pathAnswers = fileDialog.SelectedPath;
                string[] files = Directory.GetFiles(pathAnswers);
                try
                {
                    foreach (var file in files)
                    {
                        var allInfo = File.ReadAllLines(file);
                        var length = allInfo[0].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        int size = Convert.ToInt32(length[0]) * Convert.ToInt32(length[1]);
                        var infoMnist = allInfo[1].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        List<double> answerMnist = new List<double>();
                        foreach (var elem in infoMnist)
                            answerMnist.Add(Convert.ToDouble(elem));
                        answers.Add(answerMnist);
                    }
                }
                catch (System.OutOfMemoryException)
                {
                    MessageBox.Show("В выбранной папке должны быть ответы для MNIST");
                }

            }
        }

        private void trainingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form2 form = this;
            List<List<double>> answers = new List<List<double>>();
            AddAnswers(ref answers);
            neuralNetwork.Training(form, answers);
        }
    }
}
