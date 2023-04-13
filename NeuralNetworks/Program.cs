namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //var nn = new NeuralNetworks(2, 2, 2);
            //////nn.Train(new [] { 0.05, 0.1 }, new [] { 0.01, 0.09 });

            //for (int i = 0; i < 10000; i++)
            //{
            //    nn.Train(new double[] { 0.01, 0.1 }, new double[] { 0.01, 0.09 });
            //    Console.WriteLine(nn.CalculateTotalError(new double[] { 0.01, 0.1 }, new[] { 0.01, 0.09 }));
            //}

            //var c= nn.FeedForward(new double[] { 0.01, 0.1 });


            //nn.Train(new[] { 0.05, 0.1 }, new[] { 0.01, 0.09 });
            //var data = new List<double[]> { new Double[] { 1, 1 }, new Double[] { 1, 0 }, new Double[] { 0, 1 }, new Double[] { 0, 0 } };
            //var re = new double[] { 0, 1, 1, 0 };
            //nn.Train(new double[] { 0, 0 }, new[] { 0d });
            //Console.WriteLine(nn.CalculateTotalError(new double[] { 0, 0 }, new[] { 0d }));

            //var nn = new NeuralNetworks(2, 5, 1, 2);
            //for (int i = 0; i < 100000; i++)
            //{
            //    nn.Train(new double[] { 1, 1 }, new[] { 0d });
            //    nn.Train(new double[] { 1, 0 }, new[] { 1d });
            //    nn.Train(new double[] { 0, 1 }, new[] { 1d });
            //    nn.Train(new double[] { 0, 0 }, new[] { 0d });
            //    //Console.WriteLine(nn.CalculateTotalError(new double[] { 0, 0 }, new[] { 0d }));
            //}
            //var c = nn.FeedForward(new double[] { 0, 0 });
            //var c2 = nn.FeedForward(new double[] { 1, 0 });
            //var c3 = nn.FeedForward(new double[] { 0, 1 });
            //var c4 = nn.FeedForward(new double[] { 1, 1 });

            var path = Path.Combine(AppContext.BaseDirectory, "data.txt");
            var sw = new StreamReader(File.OpenRead(path));
            var tzList = new List<double[]>();
            var zlList = new List<double[]>();
            while (!sw.EndOfStream)
            {
                var line = sw.ReadLine();
                var arr = line.Split("|");
                var tzArr = arr[0].Split(",").Select(it => double.Parse(it)).ToArray();
                tzList.Add(tzArr);
                var zlArr = arr[1].Split(",").Select(it => double.Parse(it)).ToArray();
                zlList.Add(zlArr);
            }
            //2-4-201 1-5-300 1-4-301
            var nn2 = new NeuralNetworks(4, 4, 3, 2);
            for (int i = 0; i < 201; i++)
            {

                for (int j = 0; j < tzList.Count; j++)
                {
                    nn2.Train(tzList[j], zlList[j]);
                }
                //Console.WriteLine(nn.CalculateTotalError(new double[] { 0, 0 }, new[] { 0d }));
            }
            var path2 = Path.Combine(AppContext.BaseDirectory, "test.txt");
            var sw2 = new StreamReader(File.OpenRead(path2));
            var testtzList = new List<double[]>();
            var testzlList = new List<double[]>();
            while (!sw2.EndOfStream)
            {
                var line = sw2.ReadLine();
                var arr = line.Split("|");
                var tzArr = arr[0].Split(",").Select(it => double.Parse(it)).ToArray();
                testtzList.Add(tzArr);
                var zlArr = arr[1].Split(",").Select(it => double.Parse(it)).ToArray();
                testzlList.Add(zlArr);
            }
            for (int j = 0; j < testtzList.Count; j++)
            {
                var rr = nn2.FeedForward(testtzList[j]);
                var index = testzlList[j].ToList().FindIndex(it => it == 1);
                var indectIndex = rr.ToList().FindIndex(it => it == rr.Max());
                if (indectIndex != index)
                {
                    var d = 123;
                }
                Console.WriteLine($"预测为{indectIndex},实际为{index}");

            }
            var ccc = nn2.FeedForward(new double[] { 5.1, 3.5, 1.4, 0.2 });
            Console.WriteLine("Hello, World!");
        }
    }
}