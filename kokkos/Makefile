# KOKKOS_PATH = ${HOME}/Kokkos/kokkos
# KOKKOS_DEVICES = "OpenMP"
# EXE_NAME = "kokkos_jacobi"

# SRC = $(wildcard *.cpp)

# default: build
# 	echo "Start Build"


# ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
# EXE = ${EXE_NAME}.cuda
# KOKKOS_ARCH = "Volta70"
# KOKKOS_CUDA_OPTIONS = "enable_lambda"
# else
# CXX = g++
# EXE = ${EXE_NAME}.host
# KOKKOS_ARCH = "BDW"
# endif

# CXXFLAGS = -O3
# LINK = ${CXX}
# LINKFLAGS =

# DEPFLAGS = -M

# OBJ = $(SRC:.cpp=.o)
# LIB =

# include $(KOKKOS_PATH)/Makefile.kokkos

# build: $(EXE)

# $(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
# 	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

# clean: kokkos-clean
# 	rm -f *.o *.cuda *.host

# # Compilation rules

# %.o:%.cpp $(KOKKOS_CPP_DEPENDS)
# 	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

# test: $(EXE)
# 	./$(EXE)

KOKKOS_PATH = /home/noxii/Kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "kokkos_jacobi"

SRC = $(wildcard *.cpp)
OBJ_DIR = obj
TMP_DIR = tmp

OBJ = $(SRC:%.cpp=$(OBJ_DIR)/%.o)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
endif

CXXFLAGS = -O3 `pkg-config --cflags opencv4`
LINK = ${CXX}
LINKFLAGS = `pkg-config --libs opencv4`

DEPFLAGS = -M

LIB =

# Créer les répertoires de sortie s'ils n'existent pas
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TMP_DIR):
	mkdir -p $(TMP_DIR)

# Assurez-vous que le fichier Makefile.kokkos est à cet emplacement
include $(KOKKOS_PATH)/Makefile.kokkos

build: $(OBJ_DIR) $(TMP_DIR) $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(OBJ) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -rf $(OBJ_DIR) $(TMP_DIR) *.cuda *.host

# Compilation rules

$(OBJ_DIR)/%.o: %.cpp $(KOKKOS_CPP_DEPENDS) | $(OBJ_DIR)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $@

KOKKOS_TMP_DIR = $(TMP_DIR)

test: $(EXE)
	./$(EXE)
