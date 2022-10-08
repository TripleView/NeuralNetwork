namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var p1 = new double[1, 2] {{1, 1} };
            var m1 = new Matrix(p1);
            Console.WriteLine("Hello, World!");
        }
    }
}