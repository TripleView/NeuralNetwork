namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var nn = new NeuralNetworks(2, 2, 2, new double[] { 0.15, 0.2, 0.25, 0.3 }, new double[] { 0.4, 0.45, 0.5, 0.55 }, 0.35, 0.6);
            //nn.Train(new [] { 0.05, 0.1 }, new [] { 0.01, 0.09 });

            for (int i = 0; i < 10000; i++)
            {
                nn.Train(new double[] { 0.01, 0.1 }, new double[] { 0.01, 0.09 });
                Console.WriteLine(nn.CalculateTotalError(new double[] { 0.01, 0.1 }, new[] { 0.01, 0.09 }));
            }

            Console.WriteLine("Hello, World!");
        }
    }
}