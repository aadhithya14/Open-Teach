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

namespace Oculus.Interaction
{
    /// <summary>
    /// PointerCanvas allows any IPointable to forward its
    /// events onto an associated Canvas via the IPointableCanvas interface
    /// Requires a PointableCanvasModule present in the scene.
    /// </summary>
    public class PointableCanvas : PointableElement, IPointableCanvas
    {
        [SerializeField]
        private Canvas _canvas;
        public Canvas Canvas => _canvas;

        private bool _registered = false;

        private void Register()
        {
            PointableCanvasModule.RegisterPointableCanvas(this);
            _registered = true;
        }

        private void Unregister()
        {
            if (!_registered) return;
            PointableCanvasModule.UnregisterPointableCanvas(this);
            _registered = false;
        }

        protected override void OnEnable()
        {
            base.OnEnable();
            if (_started)
            {
                Register();
            }
        }

        protected override void OnDisable()
        {
            if (_started)
            {
                Unregister();
            }
            base.OnDisable();
        }
    }
}
