package inference

import (
	"context"
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
)

type tensor struct {
	shape ort.Shape
	f     []float32
	i     []int64
}
type graph struct {
	session *ort.DynamicAdvancedSession
	outputs int
}

func openGraph(path string, inputs, outputs []string, threads int) (*graph, error) {
	o, e := ort.NewSessionOptions()
	if e != nil {
		return nil, e
	}
	defer o.Destroy()
	if e = o.SetIntraOpNumThreads(threads); e != nil {
		return nil, e
	}
	if e = o.SetInterOpNumThreads(1); e != nil {
		return nil, e
	}
	s, e := ort.NewDynamicAdvancedSession(path, inputs, outputs, o)
	if e != nil {
		return nil, e
	}
	return &graph{s, len(outputs)}, nil
}
func (g *graph) close() {
	if g != nil {
		_ = g.session.Destroy()
	}
}
func (g *graph) run(ctx context.Context, input ...tensor) ([]tensor, error) {
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	in := make([]ort.Value, len(input))
	out := make([]ort.Value, g.outputs)
	defer func() {
		for _, v := range in {
			if v != nil {
				_ = v.Destroy()
			}
		}
		for _, v := range out {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()
	for k, t := range input {
		var e error
		if len(t.shape) == 0 {
			if t.f != nil {
				in[k], e = ort.NewScalar(t.f[0])
			} else {
				in[k], e = ort.NewScalar(t.i[0])
			}
		} else if t.f != nil {
			in[k], e = ort.NewTensor(t.shape, t.f)
		} else {
			in[k], e = ort.NewTensor(t.shape, t.i)
		}
		if e != nil {
			return nil, e
		}
	}
	if e := g.session.Run(in, out); e != nil {
		return nil, e
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	result := make([]tensor, len(out))
	for k, v := range out {
		result[k].shape = append(ort.Shape(nil), v.GetShape()...)
		switch t := v.(type) {
		case *ort.Tensor[float32]:
			result[k].f = append([]float32(nil), t.GetData()...)
		case *ort.Tensor[int64]:
			result[k].i = append([]int64(nil), t.GetData()...)
		case *ort.Tensor[int32]:
			for _, n := range t.GetData() {
				result[k].i = append(result[k].i, int64(n))
			}
		default:
			return nil, fmt.Errorf("unexpected ONNX output type")
		}
	}
	return result, nil
}
func ft(shape []int64, x []float32) tensor { return tensor{shape: shape, f: x} }
func it(shape []int64, x []int64) tensor   { return tensor{shape: shape, i: x} }
