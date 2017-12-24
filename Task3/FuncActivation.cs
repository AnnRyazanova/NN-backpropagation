using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task3
{
    class FuncActivation
    {
        // Линейная ф-ия
        public static double LinearFunc(double x)
        { return x; }

        // Сигмоид
        // Её диапазон значений [0,1]
        public static double SigmoidFunc(double x)
        { return 1.0 / (1.0 + Math.Exp(-1.0 * x)); }

        // Гиперболический тангенс
        // Имеет смысл использовать гиперболический тангенс, только тогда, когда значения 
        // могут быть и отрицательными, и положительными, т.к. диапазон функции [-1,1].
        public static double HyperbolicTangentFunc(double x)
        { return (Math.Exp(2 * x) - 1.0) / (Math.Exp(2 * x) + 1.0); }


        public delegate double function(double x);
        public static function GetFuncActivation(char symbol)
        {
            if (symbol.Equals('s'))
                return SigmoidFunc;
            else
                if (symbol.Equals('t'))
                return HyperbolicTangentFunc;
            else
                return LinearFunc;
        }
    }
}
