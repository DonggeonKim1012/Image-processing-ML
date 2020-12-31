#pragma once
#include "Imagelib.h"

// Tensor3D�� ũ�Ⱑ (nH x nW x nC)�� 3���� tensor�� ������

class Tensor3D {
private:
	double*** tensor;
	int nH; // height
	int nW; // width
	int nC; // channel
public:
	Tensor3D(int _nH, int _nW, int _nC) : nH(_nH), nW(_nW), nC(_nC) {
		tensor = dmatrix3D(_nH, _nW, _nC);
		for (int i = 0; i < _nH; i++)
			for (int j = 0; j < _nW; j++)
				for (int k = 0; k < _nC; k++)
					tensor[i][j][k] = 0;

		// ����: 1)3���� ����� �����Ҵ��Ͽ�, tensor�� ���� �ּҰ��� ����
		//       2)�� �� ��� element�� ���� 0���� �ʱ�ȭ
		// ����Լ�: dmatrix3D(): 3���� ����� ���� �Ҵ��ؼ� pointer�� ��ȯ�ϴ� �Լ�
	}
	~Tensor3D() {
		free_dmatrix3D(tensor, nH, nW, nC);
		// (������ ��)
		// ����: 3���� ���� �迭�� tensor�� �Ҵ� ����
		// ����Լ�: free_dmatrix3D(): 3���� ���� �Ҵ�� ����� �Ҵ� �����ϴ� �Լ�
	}
	void set_elem(int _h, int _w, int _c, double _val) { tensor[_h][_w][_c] = _val; }
	double get_elem(int _h, int _w, int _c)	const {
		return tensor[_h][_w][_c];
		// (������ ��)
		// ����: ��=_h, ��= _w, ä��= _c ��ġ element�� ��ȯ�� ��
	}

	void get_info(int& _nH, int& _nW, int& _nC) const {
		_nH = nH;
		_nW = nW;
		_nC = nC;
		// (������ ��)
		// ����: ����� ����(nH, nW, nC)�� pass by reference�� ��ȯ
	}

	void set_tensor(double*** _tensor) { tensor = _tensor; }
	double*** get_tensor() const { return tensor; }

	void print() const {
		cout << nH << '*' << nW << '*' << nC << endl;
		// (������ ��)
		// ����: ����� ũ�� (nH*nW*nC)�� ȭ�鿡 ���
	}
};