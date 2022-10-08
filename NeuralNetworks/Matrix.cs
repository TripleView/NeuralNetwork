namespace NeuralNetworks;

public class Matrix
{
    public double[,] Mat;
    public long M => Mat.GetLongLength(0);
    public long N => Mat.GetLongLength(1);

    private Random randon = new Random((int)DateTime.Now.Ticks);

    public Matrix(double[,] mat)
    {
        this.Mat=mat;
    }

    public static Matrix operator +(Matrix m1, Matrix m2)
    {
        if (m1.M != m2.M || m1.N != m2.N)
        {
            throw new Exception("2个矩阵必须有相同的行数和列数");
        }

        var result = new Matrix(new double[m1.M, m1.N]);
        for (int i = 0; i < m1.M; i++)
        {
            for (int j = 0; j < m1.N; j++)
            {
                result.Mat[i, j] = m1.Mat[i, j] + m2.Mat[i , j];
            }
        }

        return result;
    }

    public static Matrix operator *(Matrix m1, Matrix m2)
    {
        if (m1.N != m2.M)
        {
            throw new Exception("第一个矩阵的列数必须等于第二个矩阵的行数");
        }

        var result = new Matrix(new double[m1.M, m2.N]);
        for (int i = 0; i < m1.M; i++)
        {
            for (int j = 0; j < m2.N; j++)
            {
                for (int k = 0; k < m1.N; k++)
                {
                    result.Mat[i, j] += m1.Mat[i, k] * m2.Mat[k, j];
                }
            }
        }

        return result;
    }
}