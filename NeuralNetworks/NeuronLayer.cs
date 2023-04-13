namespace NeuralNetworks;

public enum NeuronLayerType
{
    隐含层=1,
    输出层=2
}

/// <summary>
/// 神经网络层
/// </summary>
public class NeuronLayer
{
    public NeuronLayerType LayerType { get; set; }
    public NeuronLayer(int neuronNums,double bias, NeuronLayerType layerType= NeuronLayerType.隐含层)
    {
        this.Neurons = new List<Neuron>();
        this.LayerType = layerType;
        for (int i = 0; i < neuronNums; i++)
        {
            this.Neurons.Add(new Neuron(bias));
        }
    }
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