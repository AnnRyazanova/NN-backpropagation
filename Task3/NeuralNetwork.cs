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

        private int maxEpoch = 5; //10; // Потолок для эпох (максимальное кол-во эпох для НС)
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

        

		//обучение АОРО, количество образцов равно количеству ответов для них
        public void Training(Form2 form, List<List<double>> answers)
        {
            int setCount = patterns.Count;
            RichTextBox richTextBox = (RichTextBox)form.Controls["richTextBox1"];

            richTextBox.AppendText("Start learning:\r\n");
            form.Refresh();

            // Если разница сумм среднеквадратичных ошибок в последних 2 эпохах меньше epsEpochError, то стоп
            double epsEpochError = 0.0001;

            // Если ошибка меньше errorEps, то стоп
            double epsError = 0.03;

            double[] lastErrors = new double[2];

			for (int i = 0; i < lastErrors.Length; ++i)
				lastErrors[i] = i;

            // Если пройдены все эпохи, то стоп
			for (int curEpoch = 0; curEpoch < maxEpoch; ++curEpoch)
			{
				bool stopLearning = false;
				double sumEpochError = 0;

				for (int iterIndex = 0; iterIndex < setCount; ++iterIndex)
				{
					List<double> input = patterns[iterIndex].Pixels;

					if (isBias)
						input.Add(1);

                    // ======================= Получение вектора ответа =======================
                    int label = patterns[iterIndex].label;
                    var labelRes = new List<double>();

                    for (int i = 0; i < 10; ++i)
                    {
                        labelRes.Add(0); // при сигмоиде
                        //labelRes.Add(-1); // при гип.тангенсе
                    }

                    labelRes[label] = 1;
                    // ========================================================================

                    List<double> output = LayerDataTransfer(input);

					double error = MSE(labelRes, output);
					sumEpochError += error;

					if (error < epsError)
						stopLearning = true;

                    string outputInfoStr = "";
                    string answerInfoStr = "";
                    foreach (var el in output)
                        outputInfoStr += Math.Round(el, 4) + " ";
                    foreach (var el in labelRes)
                        answerInfoStr += Math.Round(el, 4) + " ";
                    ChangeWeights(labelRes, output);

                    // ============================ вывод ============================
                    richTextBox.AppendText("\r\n curEpoch : " + curEpoch + "; iteration : " + iterIndex
                                            + " error = " + Math.Round(error, 4) + "\r\n");
                    richTextBox.AppendText("output : " + outputInfoStr + "\r\n");
                    richTextBox.AppendText("answer : " + answerInfoStr + "\r\n");
                    richTextBox.AppendText("label : " + label + "\r\n");
                    richTextBox.SelectionStart = richTextBox.TextLength;
                    richTextBox.ScrollToCaret();
                    form.Refresh();
                    // ================================================================
                }

                lastErrors[0] = lastErrors[1];
				lastErrors[1] = sumEpochError / setCount;

				if (Math.Abs(lastErrors[0] - lastErrors[1]) < epsEpochError)
				{
					richTextBox.AppendText("Exit : epsEpochError more then lastErrors \r\n");
					richTextBox.AppendText(lastErrors[0] + "\r\n");
					richTextBox.AppendText(lastErrors[1] + "\r\n");
					form.Refresh();
					return;
				}
				else 
                if (stopLearning)
				{
					richTextBox.AppendText("Exit : stopLearning (error < epsError)\r\n");
					form.Refresh();
					return;
				}
			}
            
			richTextBox.AppendText("Exit : passed maxEpoch\r\n");
			form.Refresh();
		}


        //Проверка корректности ответа с выходом для задач распознавания
        private bool IsCorrectAnswer(List<double> output, List<double> answer)
        {
            int patternNum = -1;

            // Поиск позиции правильного ответа в answer
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

            // Поиск позиции правильного ответа в output 
            // (Ищем большее значение сигнала)
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


        //Тест : количество верных ответов для mnist 
        public void MnistTest(Form2 form, List<List<double>> answers)
        {
            RichTextBox richTextBox = (RichTextBox)form.Controls["richTextBox1"];
            richTextBox.AppendText("\r\nStart Mnist Test :\r\n");
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
                    richTextBox.AppendText("i: " + i + " goodCount: " + goodCount + "\r\n");
                    richTextBox.SelectionStart = richTextBox.TextLength;
                    richTextBox.ScrollToCaret();
                    form.Refresh();
                }
            }

            richTextBox.AppendText("goodCount: " + goodCount + "\r\n");
            richTextBox.AppendText("Finish Mnist Test\r\n");
            richTextBox.SelectionStart = richTextBox.TextLength;
            richTextBox.ScrollToCaret();
            form.Refresh();
        }


    }
}
