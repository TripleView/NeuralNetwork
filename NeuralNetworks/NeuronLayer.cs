namespace NeuralNetworks;

/// <summary>
/// 神经网络层
/// </summary>
public class NeuronLayer
{

    public NeuronLayer(int neuronNums,double bias)
    {
        this.Bias = bias;
        this.Neurons = new List<Neuron>();

        for (int i = 0; i < neuronNums; i++)
        {
            this.Neurons.Add(new Neuron(bias));
        }
    }
    public double Bias { get; set; }
    /// <summary>
    /// 神经元
    /// </summary>
    public List<Neuron> Neurons { get; set; }

    /// <summary>
    /// 计算前馈网络
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public List<double> FeedForward(List<double> inputs)
    {
        var result = new List<double>();
        foreach (var neuron in this.Neurons)
        {
            var temp = neuron.CalculateOut(inputs);
            result.Add(temp);
        }

        return result;
    }
}