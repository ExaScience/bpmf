# Copyright (C) 2019  Jimmy Aguilar Mena

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

###############################################################################
# Find Mercurium executables in the path.
#
# This module changes the default compiler variables:
# CMAKE_CXX_COMPILER, CMAKE_Fortran_COMPILER, CMAKE_C_COMPILER. SO It
# needs to be called BEFORE calling project
#
# This also sets the following variables:
# MCXX        - C++ mercurium compiler
# MCC         - C mercurium compiler
# MFC         - FORTRAN mercurium compiler
###############################################################################

include(FindPackageHandleStandardArgs)

find_program(MCXX mcxx)
find_program(MCC mcc)
find_program(MFC mfc)

if (MCXX AND MCC AND MFC)
  set(CMAKE_CXX_COMPILER ${MCXX} CACHE INTERNAL "" FORCE)
  set(CMAKE_Fortran_COMPILER ${MFC} CACHE INTERNAL "" FORCE)
  set(CMAKE_C_COMPILER ${MCC} CACHE INTERNAL "" FORCE)
endif ()

find_package_handle_standard_args(MCC
  REQUIRED_VARS MCXX MCC MFC
  FAIL_MESSAGE "Mercurium not found in path")

