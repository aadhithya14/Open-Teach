/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
using UnityEngine.Assertions;
using Oculus.Interaction.UnityCanvas;

namespace Oculus.Interaction
{
    public class PointableCanvasMesh : PointableElement
    {
        [SerializeField]
        private CanvasRenderTextureMesh _canvasRenderTextureMesh;

        protected override void Start()
        {
            base.Start();
            Assert.IsNotNull(_canvasRenderTextureMesh);
        }

        public override void ProcessPointerEvent(PointerArgs args)
        {
            Vector3 transformPosition =
                _canvasRenderTextureMesh.ImposterToCanvasTransformPoint(args.Pose.position);
            Pose transformedPose = new Pose(transformPosition, args.Pose.rotation);
            base.ProcessPointerEvent(new PointerArgs(args.Identifier, args.PointerEvent, transformedPose));
        }

        #region Inject

        public void InjectAllCanvasMeshPointable(CanvasRenderTextureMesh canvasRenderTextureMesh)
        {
            InjectCanvasRenderTextureMesh(canvasRenderTextureMesh);
        }

        public void InjectCanvasRenderTextureMesh(CanvasRenderTextureMesh canvasRenderTextureMesh)
        {
            _canvasRenderTextureMesh = canvasRenderTextureMesh;
        }

        #endregion
    }
}
