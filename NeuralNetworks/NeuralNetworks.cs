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

    public int InputLayerNeuronNums { set; get; }
    public NeuralNetworks(int inputLayerNeuronNums, int hiddenLayerNeuronNums, int outputLayerNeuronNums, double[]? hiddenLayerWeights = null, double[]? outputLayerWeights = null, double? hiddenLayerBias = null, double? outLayerBias = null)
    {
        CheckWeight(hiddenLayerWeights);
        CheckWeight(outputLayerWeights);
        this.InputLayerNeuronNums = inputLayerNeuronNums;
        this.HiddenLayer = new NeuronLayer(hiddenLayerNeuronNums, hiddenLayerBias ?? 1);
        this.OutputLayer = new NeuronLayer(outputLayerNeuronNums, outLayerBias ?? 1);
        InitHiddenLayerNeuronWeights(hiddenLayerWeights);
        InitOutputLayerNeuronWeights(outputLayerWeights);
    }

    /// <summary>
    /// 权重不能都是0
    /// </summary>
    /// <param name="weights"></param>
    private void CheckWeight(double[]? weights)
    {
        if (weights != null)
        {
            var allTrue = true;
            for (var i = 0; i < weights.Length; i++)
            {
                var weight = weights[i];
                allTrue &= weight == 0;
            }

            if (allTrue)
            {
                throw new Exception("权重不能全部为0");
            }
        }
    }

    /// <summary>
    /// 初始化隐藏层各神经元权重
    /// </summary>
    /// <param name="hiddenLayerWeights"></param>
    private void InitHiddenLayerNeuronWeights(double[] hiddenLayerWeights)
    {
        var randon = new Random(1);
        var k = 0;
        for (var i = 0; i < this.HiddenLayer.Neurons.Count; i++)
        {
            for (int j = 0; j < InputLayerNeuronNums; j++)
            {
                var value = hiddenLayerWeights != null ? hiddenLayerWeights[k] : randon.NextDouble();
                this.HiddenLayer.Neurons[i].Weights.Add(value);
                k++;
            }
        }
    }

    /// <summary>
    /// 初始化输出层各神经元权重
    /// </summary>
    /// <param name="hiddenLayerWeights"></param>
    private void InitOutputLayerNeuronWeights(double[] outputLayerWeights)
    {
        var randon = new Random(1);
        var k = 0;
        for (var i = 0; i < this.OutputLayer.Neurons.Count; i++)
        {
            for (int j = 0; j < this.HiddenLayer.Neurons.Count; j++)
            {
                //权重必须在(0,1)之间
                var value = outputLayerWeights != null ? outputLayerWeights[k] : randon.NextDouble();
                this.OutputLayer.Neurons[i].Weights.Add(value);
                k++;
            }
        }
    }

    /// <summary>
    /// 正向传播
    /// </summary>
    public List<double> FeedForward(double[] inputs)
    {
        var hiddenLayerOutput = this.HiddenLayer.FeedForward(inputs.ToList());
        var outputLayerOutput = this.OutputLayer.FeedForward(hiddenLayerOutput);
        return outputLayerOutput;
    }

    public double CalculateTotalError(double[] inputs, double[] outputs)
    {
        this.FeedForward(inputs);
        var result = 0d;
        for (var i = 0; i < outputs.Length; i++)
        {
            var error = this.OutputLayer.Neurons[i].CalculateError(outputs[i]);
            result += error;
        }

        return result;
    }

    public void TrainHiddenLayer()
    {

    }

    public void Train(double[] inputs, double[] outputs)
    {
        this.FeedForward(inputs);

        var outCount = this.OutputLayer.Neurons.Count;
        var outListQd = new List<double>();
        for (int i = 0; i < outCount; i++)
        {
            for (int j = 0; j < this.OutputLayer.Neurons[i].Weights.Count; j++)
            {
                var c = this.OutputLayer.Neurons[i].Qd_error2Weight(outputs[i], j);
                outListQd.Add(c);

            }
          
        }

        var hiddenCount = this.HiddenLayer.Neurons.Count;
        var hiddenListQd = new List<double>();
        for (int i = 0; i < hiddenCount; i++)
        {
            var all = 0d;
            for (int j = 0; j < outCount; j++)
            {
                all += this.OutputLayer.Neurons[j].QdError2Input(outputs[j], j);
            }

            var c = all * this.HiddenLayer.Neurons[i].QdOutToWeight(i);
            hiddenListQd.Add(c);
        }


        ////输出层部分，误差对net求导
        //var outputNeuronsCount = this.OutputLayer.Neurons.Count;
        //var outputError2NetQd = Enumerable.Range(0, outputNeuronsCount).Select(it => 0d).ToList();
        //for (int i = 0; i < outputNeuronsCount; i++)
        //{
        //    outputError2NetQd[i] = this.OutputLayer.Neurons[i].Qd_error2Net(outputs[i]);
        //}


        ////隐含层部分，误差对out求导
        //var hiddenLayerNeuronsCount = this.HiddenLayer.Neurons.Count;
        //var hiddenError2outQd = Enumerable.Range(0, hiddenLayerNeuronsCount).Select(it => 0d).ToList();

        //for (int i = 0; i < hiddenLayerNeuronsCount; i++)
        //{
        //    double tempQd = 0;
        //    for (int j = 0; j < outputNeuronsCount; j++)
        //    {
        //        tempQd += outputError2NetQd[j] * this.OutputLayer.Neurons[j].Weights[i];
        //    }

        //    hiddenError2outQd[i] = tempQd * this.HiddenLayer.Neurons[i].Qd_Out2Net();
        //}

        //更新输出层
        //for (int i = 0; i < outputNeuronsCount; i++)
        //{
        //    for (int j = 0; j < this.OutputLayer.Neurons[i].Weights.Count; j++)
        //    {
        //        this.OutputLayer.Neurons[i].Weights[j] -= this.LearningRate *
        //                                                  (outputError2NetQd[i] * this.OutputLayer.Neurons[i]
        //                                                      .Qd_Net2Weight(j));
        //    }
        //}

        ////更新隐含层
        //for (int i = 0; i < hiddenLayerNeuronsCount; i++)
        //{
        //    for (int j = 0; j < this.HiddenLayer.Neurons[i].Weights.Count; j++)
        //    {
        //        this.HiddenLayer.Neurons[i].Weights[j] -= this.LearningRate *
        //                                                  (hiddenError2outQd[i] * this.HiddenLayer.Neurons[i]
        //                                                      .Qd_Net2Weight(j));
        //    }
        //}
    }
}