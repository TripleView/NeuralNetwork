namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var nn = new NeuralNetworks(2, 2, 2);
            ////nn.Train(new [] { 0.05, 0.1 }, new [] { 0.01, 0.09 });

            for (int i = 0; i < 10000; i++)
            {
                nn.Train(new double[] { 0.01, 0.1 }, new double[] { 0.01, 0.09 });
                Console.WriteLine(nn.CalculateTotalError(new double[] { 0.01, 0.1 }, new[] { 0.01, 0.09 }));
            }

            var c= nn.FeedForward(new double[] { 0.01, 0.1 });

            //var nn = new NeuralNetworks(2, 5, 1);
            //nn.Train(new [] { 0.05, 0.1 }, new [] { 0.01, 0.09 });
            // var data = new List<double[]> { new Double[] { 1, 1 }, new Double[] { 1, 0 }, new Double[] { 0, 1 }, new Double[] { 0, 0 } };
            // var re = new double[] { 0, 1, 1, 0 };
            // //nn.Train(new double[] { 0, 0 }, new[] { 0d });
            // //Console.WriteLine(nn.CalculateTotalError(new double[] { 0, 0 }, new[] { 0d }));
            // for (int i = 0; i < 100000; i++)
            // {
            //     nn.Train(new double[] { 1, 1 }, new[] { 0d });
            //     nn.Train(new double[] { 1, 0 }, new[] { 1d });
            //     nn.Train(new double[] { 0, 1 }, new[] { 1d });
            //     nn.Train(new double[] { 0, 0 }, new[] { 0d });
            //     //Console.WriteLine(nn.CalculateTotalError(new double[] { 0, 0 }, new[] { 0d }));
            // }
            //var c= nn.FeedForward(new double[]{0,0});
            //var c2 = nn.FeedForward(new double[] { 1, 0 });
            //var c3 = nn.FeedForward(new double[] {  0,1 });
            //var c4 = nn.FeedForward(new double[] { 1, 1 });
            Console.WriteLine("Hello, World!");
        }
    }
}