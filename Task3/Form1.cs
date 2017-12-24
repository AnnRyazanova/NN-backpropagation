using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Task3
{
    public partial class Form1 : Form
    {
        Form2 f2;

        public Form1()
        {
            InitializeComponent();
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            textBox1.Enabled = textBox2.Enabled = true;
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            textBox1.Enabled = textBox2.Enabled = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            int cntHidden;
            int[] cntNeurons;

            if (radioButton2.Checked)
            {
                if (!int.TryParse(textBox1.Text, out cntHidden))
                {
                    MessageBox.Show("Введите корректное кол-во скрытых слоев");
                    textBox1.Text = "";
                    return;
                }

                string[] str = textBox2.Text.Split(' ');
                if (str.Length != cntHidden)
                {
                    MessageBox.Show("Введите корректное кол-во нейронов на скрытых слоях");
                    textBox2.Text = "";
                    return;
                }

                cntNeurons = new int[cntHidden];

                for (int i = 0; i < cntHidden; ++i)
                {
                    int intParse;
                    if (!int.TryParse(str[i], out intParse))
                    {
                        MessageBox.Show("Введите корректные значения нейронов на скрытых слоях (через пробел)");
                        textBox2.Text = "";
                        return;
                    }
                    cntNeurons[i] = intParse;
                }

                f2 = new Form2(cntHidden, cntNeurons);
            }
            else
            {
                cntHidden = 1;
                cntNeurons = new int[cntHidden];
                f2 = new Form2(cntHidden, cntNeurons);
            }


            
            this.Hide();
            f2.ShowDialog();
            if (f2.DialogResult != DialogResult.OK)
                this.Visible = true;
        }
    }
}
