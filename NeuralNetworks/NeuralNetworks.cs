namespace NeuralNetworks;

/// <summary>
/// 神经网络
/// </summary>
public class NeuralNetworks
{
    /// <summary>
    /// 隐藏层
    /// </summary>
    public NeuronLayer HiddenLayer { set; get; }
    /// <summary>
    /// 输出层
    /// </summary>
    public NeuronLayer OutputLayer { get; set; }
    /// <summary>
    /// 学习率
    /// </summary>
    public double LearningRate { get; set; } = 0.5;

    public NeuralNetworks(int inputLayerNeuronNums, int hiddenLayerNeuronNums, int outputLayerNeuronNums, double[]? hiddenLayerWeights = null, double[]? outputLayerWeights = null, double? hiddenLayerBias = null, double? outLayerBias = null)
    {
        var tempHiddenLayerWeights = hiddenLayerWeights ?? Enumerable.Range(0, inputLayerNeuronNums).Select(it => 0d).ToArray();
        var tempOutputLayerWeights = outputLayerWeights ?? Enumerable.Range(0, hiddenLayerNeuronNums).Select(it => 0d).ToArray();
        this.HiddenLayer = new NeuronLayer(hiddenLayerNeuronNums, hiddenLayerBias ?? 1, tempHiddenLayerWeights.ToList());
        this.OutputLayer = new NeuronLayer(outputLayerNeuronNums, outLayerBias ?? 1, tempOutputLayerWeights.ToList());
    }

    /// <summary>
    /// 正向传播
    /// </summary>
    public void FeedForward(double[] inputs)
    {
        var hiddenLayerOutput = this.HiddenLayer.FeedForward(inputs.ToList());
        var outputLayerOutput = this.OutputLayer.FeedForward(hiddenLayerOutput);
    }

    public void Train(double[] inputs, double[] outputs)
    {
        this.FeedForward(inputs);
        //输出层部分，误差对net求导
        var outputNeuronsCount= this.OutputLayer.Neurons.Count;
        var outputError2NetQd= Enumerable.Range(0, outputNeuronsCount).Select(it => 0d).ToList();
        for (int i = 0; i < outputNeuronsCount; i++)
        {
            outputError2NetQd[i] = this.OutputLayer.Neurons[i].Qd_error2Net(outputs[0]);
        }

        //隐含层部分，误差对out求导
        var hiddenLayerNeuronsCount = this.HiddenLayer.Neurons.Count;
        var hiddenError2outQd = Enumerable.Range(0, hiddenLayerNeuronsCount).Select(it => 0d).ToList();
        
        for (int i = 0; i < hiddenLayerNeuronsCount; i++)
        {
            double tempQd = 0;
            for (int j = 0; j < outputNeuronsCount; j++)
            {
                tempQd += outputError2NetQd[j] * this.OutputLayer.Neurons[j].Weights[i];
            }

            hiddenError2outQd[i] = tempQd;
        }

        //更新输出层
        for (int i = 0; i < outputNeuronsCount; i++)
        {
            for (int j = 0; j < this.OutputLayer.Neurons[i].Weights.Count; j++)
            {
                this.OutputLayer.Neurons[i].Weights[j] -= this.LearningRate *
                                                          (outputError2NetQd[i] * this.OutputLayer.Neurons[i]
                                                              .Qd_Net2Weight(j));
            }
        }

        //更新隐含层
        for (int i = 0; i < hiddenLayerNeuronsCount; i++)
        {
            for (int j = 0; j < this.HiddenLayer.Neurons[i].Weights.Count; j++)
            {
                this.HiddenLayer.Neurons[i].Weights[j] -= this.LearningRate *
                                                          (hiddenError2outQd[i] * this.HiddenLayer.Neurons[i]
                                                              .Qd_Net2Weight(j));
            }
        }
    }
}