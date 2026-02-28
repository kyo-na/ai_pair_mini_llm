Tensor4D attention_flash4d(
    const Tensor4D& Q,
    const Tensor4D& K,
    const Tensor4D& V)
{
    int B=Q.B,T=Q.T,H=Q.H,D=Q.D;
    Tensor4D out(B,T,H,D);

    for(int b=0;b<B;++b)
    for(int h=0;h<H;++h)
    for(int t=0;t<T;++t)
    {
        float max_score=-1e9f;

        for(int tk=0;tk<=t;++tk)
        {
            float score=0;
            for(int d=0;d<D;++d)
                score+=Q.at(b,t,h,d)*K.at(b,tk,h,d);

            score/=std::sqrt((float)D);
            if(score>max_score) max_score=score;
        }

        float denom=0;

        for(int tk=0;tk<=t;++tk)
        {
            float score=0;
            for(int d=0;d<D;++d)
                score+=Q.at(b,t,h,d)*K.at(b,tk,h,d);

            score=std::exp(score/std::sqrt((float)D)-max_score);
            denom+=score;

            for(int d=0;d<D;++d)
                out.at(b,t,h,d)+=score*V.at(b,tk,h,d);
        }

        for(int d=0;d<D;++d)
            out.at(b,t,h,d)/=denom;
    }

    return out;
}