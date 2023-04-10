namespace NeuralNetworks;



/// <summary>
/// 神经元
/// </summary>
public class Neuron
{

    public Neuron(double bias)
    {
        this.Bias=bias;
        this.Weights = new List<double>();
        this.WeightQdValues = new List<double>();
    }
    /// <summary>
    /// 截距
    /// </summary>
    public double Bias { get; set; }
    /// <summary>
    /// 权重列表
    /// </summary>
    public List<double> Weights { get; set; }

    public List<double> WeightQdValues { get; set; }
 
    /// <summary>
    /// 输入列表
    /// </summary>
    public List<double> Inputs { get; set; }
    /// <summary>
    /// 输出值
    /// </summary>
    public double Output { get; set; }
    /// <summary>
    /// 计算net部分
    /// </summary>
    /// <returns></returns>
    public double CalculateNet()
    {
        var result = 0D;
        for (var i = 0; i < Weights.Count; i++)
        {
            result += Weights[i] * Inputs[i];
        }

        result += Bias;
        return result;
    }

    /// <summary>
    /// 计算out部分
    /// </summary>
    /// <returns></returns>
    public double CalculateOut(List<double> inputs)
    {
        this.Inputs = inputs;
        var netResult = CalculateNet();
        //这里选择sigmoid激活函数
        this.Output = 1 / (1 + Math.Exp(-netResult));
        return Output;
    }

    /// <summary>
    /// 计算误差
    /// </summary>
    /// <returns></returns>
    public double CalculateError(double finalResult)
    {
        return ((double)1 / 2) * Math.Pow((finalResult - this.Output),2);
    }

    /// <summary>
    /// out部分对net部分求导
    /// </summary>
    /// <returns></returns>
    public double Qd_Out2Net()
    {
        return this.Output * (1 - Output);
    }

    /// <summary>
    /// net部分对权重求导，即返回对应的input参数
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public double Qd_Net2Weight(int index)
    {
        return this.Inputs[index];
    }

    /// <summary>
    /// 误差对out部分求导
    /// </summary>
    /// <param name="errorValue"></param>
    /// <returns></returns>
    public double Qd_Error2Out(double errorValue)
    {
        return -(errorValue - this.Output);
    }


    
    /// <summary>
    /// 误差对权重求导
    /// </summary>
    /// <param name="errorValue"></param>
    /// <returns></returns>
    public void QdError2Weight(double errorValue,int index)
    {
        WeightQdValues[index] = Qd_Error2Out(errorValue) * QdOutToWeight(index);
    }

    /// <summary>
    /// 误差对权重求导
    /// </summary>
    /// <param name="errorValue"></param>
    /// <returns></returns>
    public double QdError2Input(double errorValue, int index)
    {
        return Qd_Error2Out(errorValue) * QdOutToInput(index);
    }
    public double QdOutToWeight(int index)
    {
       return this.Qd_Out2Net() * this.Inputs[index];
    }

    public double QdOutToInput(int index)
    {
        return this.Qd_Out2Net() * this.Weights[index];
    }
}