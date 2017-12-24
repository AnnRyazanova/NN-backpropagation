using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task3
{
    class ReadMnist
    {

        // mode == 0 - test
        // mode == 1 - training
        public static void ReadMnistData(out List<DigitImage> mnistInfo, int mode = 1)
        {
            int cntElem = 0;

            if (mode == 1)
                cntElem = 600; // cntElem = 60000;
            else if (mode == 0)
                cntElem = 100; // 10000

            mnistInfo = new List<DigitImage>();

            // Если режим выбран неверно
            if (cntElem == 0)
                return;

            try
            {
                FileStream ifsLabels;
                FileStream ifsImages;

                if (mode == 1)
                {
                    // train labels
                    ifsLabels = new FileStream(@"..\..\TrainingData\train-labels.idx1-ubyte", FileMode.Open);
                    // train images
                    ifsImages = new FileStream(@"..\..\TrainingData\train-images.idx3-ubyte", FileMode.Open);
                }
                else
                {
                    // test labels
                    ifsLabels = new FileStream(@"..\..\TrainingData\t10k-labels.idx1-ubyte", FileMode.Open);
                    // test images
                    ifsImages = new FileStream(@"..\..\TrainingData\t10k-images.idx3-ubyte", FileMode.Open);
                }


                BinaryReader brLabels = new BinaryReader(ifsLabels);
                BinaryReader brImages = new BinaryReader(ifsImages);


                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                //byte[][] pixels = new byte[28][];
                byte[] pixels = new byte[28 * 28];

                //for (int i = 0; i < pixels.Length; ++i)
                //    pixels[i] = new byte[28];

                // each training image
                for (int di = 0; di < cntElem; ++di)
                {
                    for (int i = 0; i < 28 * 28; ++i)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i] = b;
                    }

                    byte lbl = brLabels.ReadByte();
                    
                    mnistInfo.Add(new DigitImage(pixels, lbl));
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();


            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        }
    

        public class DigitImage
        {
            private List<byte> pixels;
            public byte label;

            public List<double> Pixels { get { return pixels.ConvertAll(x => (double)x); } }

            public DigitImage(byte[] pixels, byte label)
            {
                this.pixels = new List<byte>(28 * 28); ;

                //for (int i = 0; i < this.pixels.Length; ++i)
                //    this.pixels[i] = new byte[28];

                //for (int i = 0; i < 28; ++i)
                //    for (int j = 0; j < 28; ++j)
                //        this.pixels[i][j] = pixels[i][j];

                for (int i = 0; i < 28 * 28; ++i)
                    this.pixels.Add(pixels[i]);

                this.label = label;
            }

            



        }
    }
}
