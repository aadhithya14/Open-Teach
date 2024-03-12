/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;

namespace Oculus.Interaction.UnityCanvas
{
    public class CanvasRect : CanvasRenderTextureMesh
    {
        protected override OVROverlay.OverlayShape OverlayShape => OVROverlay.OverlayShape.Quad;

        protected override void UpdateOverlayPositionAndScale()
        {
            if (_overlay == null)
            {
                return;
            }

            var resolution = _canvasRenderTexture.GetBaseResolutionToUse();
            _overlay.transform.localPosition = -_runtimeOffset;
            _overlay.transform.localScale = new Vector3(_canvasRenderTexture.PixelsToUnits(resolution.x),
                                                        _canvasRenderTexture.PixelsToUnits(resolution.y),
                                                        1);
        }

        protected override Vector3 MeshInverseTransform(Vector3 localPosition)
        {
            return localPosition;
        }

        protected override void GenerateMesh(out List<Vector3> verts,
                                             out List<int> tris,
                                             out List<Vector2> uvs)
        {
            verts = new List<Vector3>();
            tris = new List<int>();
            uvs = new List<Vector2>();

            var resolution = _canvasRenderTexture.GetBaseResolutionToUse();
            Vector2 worldSize = new Vector2(
                _canvasRenderTexture.PixelsToUnits(Mathf.RoundToInt(resolution.x)),
                _canvasRenderTexture.PixelsToUnits(Mathf.RoundToInt(resolution.y))
                ) / transform.lossyScale;

            float xPos = worldSize.x * 0.5f;
            float xNeg = -xPos;

            float yPos = worldSize.y * 0.5f;
            float yNeg = -yPos;

            verts.Add(new Vector3(xNeg, yNeg, 0));
            verts.Add(new Vector3(xNeg, yPos, 0));
            verts.Add(new Vector3(xPos, yPos, 0));
            verts.Add(new Vector3(xPos, yNeg, 0));

            tris.Add(0);
            tris.Add(1);
            tris.Add(2);

            tris.Add(0);
            tris.Add(2);
            tris.Add(3);

            uvs.Add(new Vector2(0, 0));
            uvs.Add(new Vector2(0, 1));
            uvs.Add(new Vector2(1, 1));
            uvs.Add(new Vector2(1, 0));
        }

        #region Inject

        public void InjectAllCanvasRect(CanvasRenderTexture canvasRenderTexture)
        {
            InjectAllCanvasRenderTextureMesh(canvasRenderTexture);
        }

        #endregion
    }
}
