using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task3
{
    class NNLayer
    {
        static private Random rand = new Random();
       
        public List<double> weight = new List<double>();
        public List<double> outputs = new List<double>(); 
        public FuncActivation.function func;
        public int cntNeurons;
        public int cntWeights;

        public NNLayer(int _cntNeurons, int sizePrevLayer, char symbol)
        {
            cntNeurons = _cntNeurons;
            func = FuncActivation.GetFuncActivation(symbol);
            outputs = new List<double>(cntNeurons);

            if (sizePrevLayer > 0) // Если это не входной слой НС
            {
                cntWeights = cntNeurons * sizePrevLayer;
                for (int i = 0; i < cntWeights; ++i)
                    weight.Add(GetRandomWeight());
            }
        }


        // Генерация рандомного веса в заданных промежутках
        // Здесь генерируется число в пределах от a = -0.4 до b = 0.4
        // Формула : (b - a) * rand + a;
        private double GetRandomWeight()
        {
            return 0.8 * rand.NextDouble() - 0.4;
        }


    }
}
