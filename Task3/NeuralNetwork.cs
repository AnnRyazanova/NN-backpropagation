using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Task3
{
    class NeuralNetwork
    {
        // Есть ли нейроны смещения (на входе и выходе нейрон имеет вес, равный 1)
        private bool isBias = false;

        private List<ReadMnist.DigitImage> patterns;
        private List<ReadMnist.DigitImage> tests;

        private int maxEpoch = 10; // Потолок для эпох (максимальное кол-во эпох для НС)
        private int curEpoch = 0; // Текущая эпоха
        private int trainSetSize; // Размер тренировочного/тестового сета
        private double learningRate = 0.5; // Скорость обучения

        // Слои:
        // 1. входной слой
        private NNLayer input;
        // 2. скрытые слои
        private List<NNLayer> hidden = new List<NNLayer>(); 
        // 3. выходной слой
        private NNLayer output;

        
        // Общее кол-во слоев НС
        private int cntLayers;
        
        // НС по умолчанию с 1 скрытым слоем
        public NeuralNetwork(List<ReadMnist.DigitImage> _patterns, int _cntHiddenLayers, int[] _cntNeuronsOnHiddenLayers, List<ReadMnist.DigitImage> _tests)
        {
            int outLayerSize; // Размер выходного слоя 
            int sizeLayer; // Размер текущего слоя (кол-во нейронов на нем);

            cntLayers = 2 + _cntHiddenLayers;
            patterns = _patterns;
            tests = _tests;
            outLayerSize = sizeLayer = trainSetSize = patterns[0].Pixels.Count;
            
            if (isBias)
                sizeLayer++;

            input = new NNLayer(sizeLayer, 0, 'l');
            hidden.Add(new NNLayer(sizeLayer, sizeLayer, 's'));
            output = new NNLayer(10, sizeLayer, 's');
        }

        // НС с выбором основных параметров: число скрытых слоев, количество нейронов на каждом слое
        public NeuralNetwork(List<ReadMnist.DigitImage> _patterns, int[] cntNeuronsOnHiddenLayers, int outSize, List<ReadMnist.DigitImage> _tests)
        {
            patterns = _patterns;
            tests = _tests;
            int outLayerSize = outSize;
            int inputSize = patterns.Count;

            if (isBias)
                inputSize++;

            input = new NNLayer(inputSize, 0, 'l');
            int prevSize = inputSize;

            foreach (int size in cntNeuronsOnHiddenLayers)
            {
                int sizeLayer = size;

                if (isBias)
                    sizeLayer++;

                hidden.Add(new NNLayer(sizeLayer, prevSize, 's'));
                prevSize = sizeLayer;
            }
            output = new NNLayer(outLayerSize, prevSize, 's');
            
            cntLayers = 2 + hidden.Count;
        }

        // По значениям предыдущего слоя получаем вход для следующего (Сумма произведений сигнал * вес)
        // Манипуляции с весами, сигналами и функцией активации
        private List<double> FromInputToOutputOnNeurons(List<double> inputInfo, NNLayer nextLayer)
        {
            List<double> outputInfo = new List<double>(nextLayer.cntNeurons);
            
            for (int i = 0; i < nextLayer.cntNeurons; ++i)
            {
                double sum = 0;
                int lastInputIndex = inputInfo.Count - 1;
                for (int j = 0; j < lastInputIndex; ++j)
                {
                    double signal = inputInfo[j];
                    sum += signal * nextLayer.weight[i * inputInfo.Count + j];
                }

                // Если есть нейрона смещения, то добавляем вес нейрона смещения, 
                // иначе - вес последнего нейрона в слое
                sum += (isBias ? 1 : inputInfo[lastInputIndex]) * 
                    nextLayer.weight[i * inputInfo.Count + lastInputIndex];
                
                outputInfo.Add(nextLayer.func(sum));
            }
            return outputInfo;
        }


        // Передача данных по слоям : выход предыдущего слоя == входу текущего
        public List<double> LayerDataTransfer(List<double> inputInfo)
        {
            // Выход входного слоя совпадаем с самим входным сигналом
            input.outputs = inputInfo;
            List<double> curInputInfo = inputInfo;

            // Работа со скрытыми слоями :
            // Выход входного слоя == входом первого скрытого слоя
            // Выход первого скрытого слоя == входом второго скрытого слоя и т.д.
            foreach (var curLayer in hidden)
            {
                curInputInfo = FromInputToOutputOnNeurons(curInputInfo, curLayer);
                if (isBias)
                    curInputInfo[curInputInfo.Count - 1] = 1;
                curLayer.outputs = curInputInfo;
            }

            // Работа с выходным слоем : 
            output.outputs = curInputInfo = FromInputToOutputOnNeurons(curInputInfo, output);
            return curInputInfo;
        }

        // Производная функции активации
        private double GetDerivativeFuncActivation(char symbol, double outputInfo)
        {
            if (symbol.Equals('s'))
                return (1 - outputInfo) * outputInfo; // сигмоид
            else
                return 1 - Math.Pow(outputInfo, 2); // гиперболический тангенс
        }

        // Изменение весов 
        // Метод обратного распространения ошибки (один обратный проход с изменением весов и подсчетом дельт)
        private void ChangeWeights(List<double> idealOutput, List<double> outputInfo)
        {
            NNLayer nextLayer = output;
            List<double> deltaNext = new List<double>(idealOutput.Count);

            // Производная ф-ии активации от входного значения нейрона
            double fin = 0;

            // Находим дельту для выходного слоя
            // delta = (OUTideal - OUTcurrent) * fin
            for (int i = 0; i < idealOutput.Count; ++i)
            {
                // Производная ф-ии активации (здесь : гиперболический тангенс)
                fin = GetDerivativeFuncActivation('s', outputInfo[i]); 
                deltaNext.Add((idealOutput[i] - outputInfo[i]) * fin);
            }

            // Находим дельты для скрытых слоев
            for (int i = hidden.Count - 1; i >= -1; --i)
            {
                NNLayer currentLayer;
                currentLayer = i != -1 ? hidden[i] : input;
                List<double> newWeights = new List<double> (nextLayer.weight.Count);
                int index = 0;

                // пересчет веса (на каждой связи нейронов текущего слоя и нейронов следующего слоя)
                for (int k = 0; k < nextLayer.cntNeurons; ++k)
                {
                    for (int j = 0; j < currentLayer.cntNeurons; ++j)
                    {
                        // градиент j, k
                        double grad = deltaNext[k] * currentLayer.outputs[j];
                        // дельта веса
                        double deltaWeight = learningRate * grad;
                        newWeights.Add(nextLayer.weight[index++] + deltaWeight);
                    }
                }
                // ---

                List<double> deltaCurrent = new List<double>(deltaNext);
                deltaNext.Clear();

                // Подстчет дельт данного слоя
                // delta = (fin * sum(W[i] * Delta[i]))
                for (int j = 0; j < currentLayer.cntNeurons; ++j)
                {
                    double sum = 0;
                    for (int k = 0; k < nextLayer.cntNeurons; ++k)
                        sum += nextLayer.weight[currentLayer.cntNeurons * k + j] * deltaCurrent[k];

                    // Производная ф-ии активации (здесь : гиперболический тангенс)
                    fin = GetDerivativeFuncActivation('s', currentLayer.outputs[j]);
                    deltaNext.Add(fin * sum);
                }
                for (int wI = 0; wI < nextLayer.cntWeights; ++wI)
                    nextLayer.weight[wI] = newWeights[wI];
                nextLayer = currentLayer;
                // ---

            }

        }

		//  Mean Squared Error
		// [(ideal[1] - current[1])^2 + ... + (ideal[n] - current[n])^2] / n
		// n - количество сетов
		private double MSE(List<double> ideal, List<double> current)
		{
			int n = ideal.Count;
			double error = 0;
			for (int i = 0; i < n; ++i)
				error += Math.Pow((ideal[i] - current[i]), 2);
			error = error / n;
			return error;
		}

		//обучение МОР, количество образцов равно количеству ответов для них
		private int DIST_PRINT = 1;//печать через
        public void Training(Form2 form, List<List<double>> answers)
        {
            int setCount = patterns.Count;
            RichTextBox richTextBox = (RichTextBox)form.Controls["richTextBox1"];

            richTextBox.AppendText("Start learning:\r\n");
            form.Refresh();
            double epsEpErr = 0.0001f;
            double[] lastErrors = new double[3];

			for (int i = 0; i < lastErrors.Length; ++i)
				lastErrors[i] = i;

			for (int ep = 0; ep < maxEpoch; ++ep)
			{
				bool isStopLearning = true;
				double sumEpErr = 0;

				for (int setIter = 0; setIter < setCount; ++setIter)
				{
					List<double> input = patterns[setIter].Pixels;

					if (isBias)
						input.Add(1);

                    int label = patterns[setIter].label;
                    var labelRes = new List<double>();

                    for (int i = 0; i < 10; ++i)
                        labelRes.Add(0); // т.к. гип.тангенс, 0 при сигмоиде

                    labelRes[label] = 1;

					List<double> output = LayerDataTransfer(input);

					double err = MSE(labelRes, output);
					sumEpErr += err;

					if (err > 0.03)
						isStopLearning = false;

                    string outStr = "";
                    string answStr = "";
                    foreach (var el in output)
                        outStr += Math.Round(el, 4) + " ";
                    foreach (var el in answers[label])
                        answStr += Math.Round(el, 4) + " ";
                    ChangeWeights(answers[label], output);

                    // ============================ вывод ============================
                    richTextBox.AppendText("ep = " + ep + " setIter = " + setIter
                            + " err = " + Math.Round(err, 4) + "\r\n");
                    richTextBox.AppendText("out: " + outStr + "\r\n");
                    richTextBox.AppendText("answ: " + answStr + "\r\n");
                    richTextBox.AppendText("label = " + label + "\r\n");
                    richTextBox.AppendText(patterns[setIter].ToString());
                    richTextBox.SelectionStart = richTextBox.TextLength;
                    richTextBox.ScrollToCaret();
                    form.Refresh();
                    // ================================================================
                }

                lastErrors[0] = lastErrors[1];
				lastErrors[1] = lastErrors[2];
				lastErrors[2] = sumEpErr / setCount;

				if (Math.Abs(lastErrors[0] - lastErrors[1]) < epsEpErr ||
					Math.Abs(lastErrors[1] - lastErrors[2]) < epsEpErr)
				{
					richTextBox.AppendText("exit epsEpErr\r\n");
					richTextBox.AppendText(lastErrors[0] + "\r\n");
					richTextBox.AppendText(lastErrors[1] + "\r\n");
					richTextBox.AppendText(lastErrors[2] + "\r\n");
					form.Refresh();
					return;
				}
				if (isStopLearning)
				{
					richTextBox.AppendText("exit isStopLearning\r\n");
					form.Refresh();
					return;
				}
			}
			richTextBox.AppendText("exit MAX_EP\r\n");
			form.Refresh();

		}


        //Проверка корректности ответа с выходом для задач распознавания
        private bool IsCorrectAnswer(List<double> output, List<double> answer)
        {
            int patternNum = -1;
            for (int i = 0; i < answer.Count; ++i)
            {
                if (answer[i] == 1)
                {
                    patternNum = i;
                    break;
                }
            }
            double maxOutSignal = double.MinValue;
            int maxOutSignalPos = -1;
            for (int i = 0; i < output.Count; ++i)
            {
                if (maxOutSignal < output[i])
                {
                    maxOutSignal = output[i];
                    maxOutSignalPos = i;
                }
            }
            return maxOutSignalPos == patternNum;
        }



        //Тест количество верных ответов для mnist TestDigits
        public void MnistTest(Form2 form, List<List<double>> answers)
        {
            RichTextBox richTextBox = (RichTextBox)form.Controls["richTextBox1"];
            richTextBox.AppendText("\r\nstart Mnist Test\r\n");
            richTextBox.SelectionStart = richTextBox.TextLength;
            richTextBox.ScrollToCaret();
            form.Refresh();
            
            int goodCount = 0;
            for (int i = 0; i < tests.Count; ++i)
            {
                List<double> input = tests[i].Pixels;
                if (isBias)
                    input.Add(1);
                int label = tests[i].label;
                List<double> output = LayerDataTransfer(input);
                if (IsCorrectAnswer(output, answers[label]))
                    goodCount++;
                if (i % 100 == 0)
                {
                    richTextBox.AppendText("i = " + i + " goodCount = " + goodCount + "\r\n");
                    richTextBox.SelectionStart = richTextBox.TextLength;
                    richTextBox.ScrollToCaret();
                    form.Refresh();
                }
            }

            richTextBox.AppendText("goodCount = " + goodCount + "\r\nfinish Mnist Test" + "\r\n");
            richTextBox.SelectionStart = richTextBox.TextLength;
            richTextBox.ScrollToCaret();
            form.Refresh();
        }


    }
}
