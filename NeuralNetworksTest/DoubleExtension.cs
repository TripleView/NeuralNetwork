namespace NeuralNetworksTest;

public static class DoubleExtension
{
    public static bool ValueEqual(this double[,] m1, double[,] m2)
    {
        if (m1.GetLongLength(0) != m2.GetLongLength(0) || m1.GetLongLength(1) != m2.GetLongLength(1))
        {
            return false;
        }

        for (int i = 0; i < m1.GetLongLength(0); i++)
        {
            for (int j = 0; j < m1.GetLongLength(1); j++)
            {
                if (m1[i, j] != m2[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }
}