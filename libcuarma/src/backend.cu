#pragma once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

// include necessary system headers
#include <iostream>

#include "cuarma.hpp"
#include "cuarma_private.hpp"


CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaBackendCreate(cuarmaBackend * backend)
{
  *backend = new cuarmaBackend_impl();

  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaBackendDestroy(cuarmaBackend * backend)
{
  delete *backend;
  *backend = NULL;

  return cuarmaSuccess;
}

