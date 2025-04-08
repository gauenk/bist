/********************************************************************

      Pairwise differences between pixels and superpixels

********************************************************************/

torch::Tensor sparse_pwd_py(torch::Tensor video,
                            torch::Tensor down,
                            torch::Tensor spix_read);
