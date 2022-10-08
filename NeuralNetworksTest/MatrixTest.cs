using NeuralNetworks;

namespace NeuralNetworksTest
{
    public class MatrixTest
    {
        [Fact]
        public void TestNeuralNetworks()
        {
            var nn = new NeuralNetworks.NeuralNetworks(2, 2, 2,new double[] {0.15, 0.2},new double[]{ 0.4, 0.45 },0.35,0.6);
            for (int i = 0; i < 10000; i++)
            {
                nn.Train(new double[]{ 0.01, 0.1 }, new double[] { 0.01, 0.09 });
            }

            var aa = 123;
        }


        
        [Fact]
        public void TestEnumerableRange()
        {
            var a = Enumerable.Range(0, 2).Select(it => it).ToList();
        }

        [Fact]
        public void TestMatricAdd()
        {
            var m1 = new Matrix(new double[2, 2] { { 1, 1 }, { 2, 2 } });
            var m2 = new Matrix(new double[2, 2] { { 3, 3 }, { 4, 4 } });
            var result = m1 + m2;
            Assert.True(result.Mat.ValueEqual(new double[2, 2] { { 4, 4 }, { 6, 6 } }));
        }

        [Fact]
        public void TestMatricMulti2()
        {
            var m1 = new Matrix(new double[3, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var m2 = new Matrix(new double[2, 1] { { 4 }, { 5 } });
            var result = m1 * m2;
            Assert.True(result.Mat.ValueEqual(new double[3, 3] { { 15, 18, 21 }, { 33, 40, 47 }, { 51, 62, 73 } }));
        }

        [Fact]
        public void TestMatricMulti()
        {
            var m1 = new Matrix(new double[3, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var m2 = new Matrix(new double[2, 3] { { 3, 4, 5 }, { 6, 7, 8 } });
            var result = m1 * m2;
            Assert.True(result.Mat.ValueEqual(new double[3, 3] { { 15, 18, 21 }, { 33, 40, 47 }, { 51, 62, 73 } }));
        }
    }
}