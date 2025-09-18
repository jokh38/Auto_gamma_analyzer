## MCC 좌표계 분석 요약표

### 현재 코드 상태 (`+1` 있음)

| 구분 | 좌표 생성/변환 방식 | 사용되는 곳 | 결과 | 정확성 |
|------|-------------------|------------|------|--------|
| **phys_x_mesh, phys_y_mesh** | `(index - origin + 1) × spacing` | 2D 맵 표시 (extent) | 인덱스 25 → 0mm<br>인덱스 26 → +5mm | ✗ (1픽셀 shift) |
| **pixel_to_physical_coord()** | `(index - origin) × spacing` | 감마 분석 좌표 계산 | 인덱스 26 → 0mm | ✓ (정확) |
| **2D 맵 표시** | `extent = [phys_x_mesh.min/max]` | imshow의 extent 파라미터 | +1 픽셀 shift된 extent | - |
| **감마 맵 표시** | `extent = [phys_x_mesh.min/max]` | imshow의 extent 파라미터 | +1 픽셀 shift된 extent | - |
| **감마 계산** | `(index - origin) × spacing` | 실제 감마값 계산 | 정확한 물리 좌표 | ✓ |
| **감마 맵 매핑** | 원본 MCC 인덱스에 직접 저장 | `gamma_map[original_indices]` | 2D 맵과 동일 위치 | ✓ (상대적 일치) |
| **1D 프로파일 위치 찾기** | `find_nearest(phys_x_mesh, position)` | 프로파일 추출 인덱스 | 인덱스 25 선택 (0mm 위치) | ✗ (실제 26이어야) |
| **1D 프로파일 좌표** | `phys_x_mesh[indices]` 직접 사용 | 프로파일 x축 좌표 | +1 픽셀 shift | ✗ |

### `+1` 제거 시

| 구분 | 좌표 생성/변환 방식 | 사용되는 곳 | 결과 | 정확성 |
|------|-------------------|------------|------|--------|
| **phys_x_mesh, phys_y_mesh** | `(index - origin) × spacing` | 2D 맵 표시 (extent) | 인덱스 26 → 0mm | ✓ (정확) |
| **pixel_to_physical_coord()** | `(index - origin) × spacing` | 감마 분석 좌표 계산 | 인덱스 26 → 0mm | ✓ (정확) |
| **2D 맵 표시** | `extent = [phys_x_mesh.min/max]` | imshow의 extent 파라미터 | 정확한 extent | ✓ |
| **감마 맵 표시** | `extent = [phys_x_mesh.min/max]` | imshow의 extent 파라미터 | 정확한 extent | ✓ |
| **감마 계산** | `(index - origin) × spacing` | 실제 감마값 계산 | 정확한 물리 좌표 | ✓ |
| **감마 맵 매핑** | 원본 MCC 인덱스에 직접 저장 | `gamma_map[original_indices]` | 2D 맵과 동일 위치 | ✓ |
| **1D 프로파일 위치 찾기** | `find_nearest(phys_x_mesh, position)` | 프로파일 추출 인덱스 | 인덱스 26 선택 (0mm 위치) | ✓ |
| **1D 프로파일 좌표** | `phys_x_mesh[indices]` 직접 사용 | 프로파일 x축 좌표 | 정확한 좌표 | ✓ |

## 시각적 일치/불일치 분석

### 현재 상태 (`+1` 있음)
```
실제 데이터 위치:     [인덱스 26 = 0mm]
2D 맵 표시:          [인덱스 25 = 0mm] ← 1픽셀 shift
감마 맵 표시:        [인덱스 25 = 0mm] ← 1픽셀 shift  
감마 계산 위치:      [인덱스 26 = 0mm] ← 정확
감마 맵에 저장:      [인덱스 26에 저장] → 2D와 같은 위치에 표시됨 ✓

1D 프로파일:
- 0mm 찾기 → 인덱스 25 선택 (잘못된 위치)
- 표시 좌표 → +5mm로 표시 (shift 발생) ✗
```

### `+1` 제거 시
```
실제 데이터 위치:     [인덱스 26 = 0mm]
2D 맵 표시:          [인덱스 26 = 0mm] ← 정확
감마 맵 표시:        [인덱스 26 = 0mm] ← 정확
감마 계산 위치:      [인덱스 26 = 0mm] ← 정확
감마 맵에 저장:      [인덱스 26에 저장] → 모두 일치 ✓

1D 프로파일:
- 0mm 찾기 → 인덱스 26 선택 (정확한 위치)
- 표시 좌표 → 0mm로 표시 (정확) ✓
```

## 결론

**현재 코드**는 2D 표시를 위해 `+1` 오프셋을 사용했지만, 이것이 실제로는 부정확한 좌표계입니다. 감마 분석 결과가 정렬되어 보이는 것은 **둘 다 같은 방식으로 shift되어 있기 때문**입니다.

**권장 해결책**: `+1` 제거
- 모든 좌표계가 일관되고 정확해짐
- 1D, 2D 프로파일, 감마 분석 모두 올바른 물리 좌표 사용